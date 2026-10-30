# SPDX-License-Identifier: GPL-3.0-or-later
"""Export side, step 2 (pure Python): :class:`nif_scene.ActionDesc` -> Oblivion animation blocks.

Layout (mirrors vanilla doors / fountains / glowing pieces)::

    root.Controller -> NiControllerManager -> NiMultiTargetTransformController (Extra Targets = animated nodes)
    NiControllerManager.Controller Sequences -> NiControllerSequence (one per Blender action group)
        Controlled Blocks -> interpolator (+ data), controller, string-palette offsets
    NiControllerManager.Object Palette -> NiDefaultAVObjectPalette (every named AV object)
    material / alpha / visibility controllers sit on their NiMaterialProperty / shape with an
    empty NiBlend*Interpolator, exactly like the vanilla files.
"""
import numpy as np

from .nif_anim import mat_to_quat
from .nif_scene import bone_axis_matrix, transform_matrix

FLT_MAX = 3.4028234663852886e+38
INVALID = -FLT_MAX
NO_OFFSET = 0xFFFFFFFF
CYCLE = {'LOOP': 0, 'REVERSE': 1, 'CLAMP': 2}
COLOR_SLOT = {'ambient': (0, 'AMBIENT'), 'diffuse': (1, 'DIFFUSE'), 'specular': (2, 'SPECULAR'), 'emissive': (3, 'SELF_ILLUM')}
KEY_LINEAR, KEY_QUADRATIC, KEY_CONST = 1, 2, 5


# ------------------------------------------------------------------ key groups
def _vec(v):
    return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}


def keygroup(track, comp=None, k=1, vscale=1.0, force=None):
    """KeyTrack -> KeyGroup dict.  ``k`` = 1 (float) or 3 (Vector3); ``comp`` selects the column for k == 1."""
    if track is None or len(track.times) == 0:
        return {'Num Keys': 0, 'Keys': []}
    n = len(track.times)
    mode = force or track.interp
    if mode == 'BEZIER' and track.fwd is None:
        mode = 'LINEAR'
    code = {'LINEAR': KEY_LINEAR, 'BEZIER': KEY_QUADRATIC, 'CONSTANT': KEY_CONST}[mode]

    def val(a, i):
        return float(a[i, comp or 0]) * vscale if k == 1 else _vec(a[i, :3] * vscale)

    keys = []
    for i in range(n):
        key = {'Time': float(track.times[i]), 'Value': val(track.values, i)}
        if code == KEY_QUADRATIC:
            key['Forward'] = val(track.fwd, i)
            key['Backward'] = val(track.bwd, i)
        keys.append(key)
    return {'Num Keys': n, 'Interpolation': code, 'Keys': keys}


def bool_keygroup(track):
    keys = [{'Time': float(t), 'Value': int(v[0] > 0.5)} for t, v in zip(track.times, track.values)]
    return {'Num Keys': len(keys), 'Interpolation': KEY_CONST, 'Keys': keys}


def transform_data_fields(loc, quat, euler, scale, loc_scale, start, stop):
    d = {}
    if quat is not None:
        qk = [{'Time': float(t), 'Value': {'w': float(q[0]), 'x': float(q[1]), 'y': float(q[2]), 'z': float(q[3])}}
              for t, q in zip(quat.times, quat.values)]
        d.update({'Num Rotation Keys': len(qk), 'Rotation Type': KEY_CONST if quat.interp == 'CONSTANT' else KEY_LINEAR,
                  'Quaternion Keys': qk})
    elif euler is not None:
        groups = []
        for tr in euler:
            if tr is None:                                    # static axis: two equal keys like vanilla files
                groups.append(None)
            else:
                groups.append(keygroup(tr, 0, 1))
        d.update({'Num Rotation Keys': 1, 'Rotation Type': 4, 'XYZ Rotations': groups})
    else:
        d['Num Rotation Keys'] = 0
    d['Translations'] = keygroup(loc, None, 3, loc_scale)
    d['Scales'] = keygroup(scale, 0, 1)
    return d


# -------------------------------------------------------------------- exporter
class AnimExporter:
    def __init__(self, builder, root, actions):
        self.b = builder
        self.root = root
        self.actions = actions
        self.strings = ''                        # string palette text
        self.offsets = {}
        self.ctrl_cache = {}                     # (id(block), ctype, slot) -> controller block
        self.targets = []                        # transform-animated node blocks
        self.warn = builder.warnings

    # ---- helpers
    def s(self, text):
        if text is None:
            return NO_OFFSET
        if text not in self.offsets:
            self.offsets[text] = len(self.strings)
            self.strings += text + '\x00'
        return self.offsets[text]

    def _chain_attach(self, owner, ctrl):
        cur = int(owner['Controller'])
        if cur < 0:
            owner['Controller'] = ctrl.index
            return
        blk = self.b.blocks[cur]
        while int(blk['Next Controller']) >= 0:
            blk = self.b.blocks[int(blk['Next Controller'])]
        blk['Next Controller'] = ctrl.index

    @staticmethod
    def _time_fields(start, stop, flags):
        return {'Next Controller': -1, 'Flags': flags, 'Frequency': 1.0, 'Phase': 0.0,
                'Start Time': float(start), 'Stop Time': float(stop)}

    def _node_rest(self, blk):
        A = blk['Rotation']
        R = np.array([[A['m11'], A['m21'], A['m31']], [A['m12'], A['m22'], A['m32']], [A['m13'], A['m23'], A['m33']]])
        t = blk['Translation']
        return np.array([t['x'], t['y'], t['z']]), R, float(blk['Scale'])

    # ---- transform interpolators
    def transform_interp(self, blk, loc, quat, euler, scale, t0, t1):
        b = self.b
        t, R, sc = self._node_rest(blk)
        rq = mat_to_quat(R)
        has_rot = quat is not None or euler is not None
        data_fields = transform_data_fields(loc, quat, euler, scale, b.inv, t0, t1)
        if euler is not None:                                 # static axes: constant keys
            ang = _euler_of(R)
            groups = []
            for i, g in enumerate(data_fields['XYZ Rotations']):
                if g is None:
                    g = {'Num Keys': 2, 'Interpolation': KEY_LINEAR, 'Keys': [
                        {'Time': float(t0), 'Value': float(ang[i])}, {'Time': float(t1), 'Value': float(ang[i])}]}
                groups.append(g)
            data_fields['XYZ Rotations'] = groups
        data = b.add('NiTransformData', **data_fields)
        tr = {'Translation': _vec([INVALID] * 3) if loc is not None else _vec(t),
              'Rotation': ({'w': INVALID, 'x': INVALID, 'y': INVALID, 'z': INVALID} if has_rot else
                           {'w': float(rq[0]), 'x': float(rq[1]), 'y': float(rq[2]), 'z': float(rq[3])}),
              'Scale': INVALID if scale is not None else float(sc)}
        return b.add('NiTransformInterpolator', **{'Transform': tr, 'Data': data.index})

    def bone_tracks(self, arm, bone, ch):
        """BONE channel (pose basis, Blender units) -> NIF local loc/quat/scale tracks sampled per key."""
        from .nif_scene import KeyTrack
        b = self.b
        blk = b.bone_blocks.get((id(arm), bone))
        if blk is None:
            return None, None
        s = b.s.scale
        C = np.eye(4)
        C[:3, :3] = bone_axis_matrix(int(arm.props.get('nif_bone_forward', 0)))
        Ci = np.linalg.inv(C)
        rest = transform_matrix(blk)                          # parent-local NIF rest
        rest_b = rest.copy()
        rest_b[:3, 3] *= s
        n = len(ch.loc.times)
        L, Q, S = np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, 1))
        for i in range(n):
            P = np.eye(4)
            from .nif_anim import quat_to_mat
            P[:3, :3] = quat_to_mat(ch.quat.values[i]) * ch.scale.values[i]
            P[:3, 3] = ch.loc.values[i]
            Lt = rest_b @ C @ P @ Ci
            sc = float(np.cbrt(abs(np.linalg.det(Lt[:3, :3]))) or 1.0)
            L[i] = Lt[:3, 3]
            q = mat_to_quat(Lt[:3, :3] / sc)
            if i and q @ Q[i - 1] < 0:
                q = -q
            Q[i] = q
            S[i, 0] = sc
        T = ch.loc.times
        return blk, (KeyTrack(T, L), KeyTrack(T, Q), KeyTrack(T, S))

    # ---- placeholders / controllers
    def blend(self, kind):
        if kind == 'point3':
            return self.b.add('NiBlendPoint3Interpolator', Flags=1, **{'Array Size': 2, 'Weight Threshold': 0.0,
                                                                        'Value': _vec([INVALID] * 3)})
        if kind == 'bool':
            return self.b.add('NiBlendBoolInterpolator', Flags=1, **{'Array Size': 2, 'Weight Threshold': 0.0, 'Value': 2})
        return self.b.add('NiBlendFloatInterpolator', Flags=1, **{'Array Size': 2, 'Weight Threshold': 0.0, 'Value': INVALID})

    def property_controller(self, owner, ctype, kind, stop, extra=None):
        key = (id(owner), ctype, (extra or {}).get('Target Color'))
        if key in self.ctrl_cache:
            return self.ctrl_cache[key]
        f = self._time_fields(0.0, stop, 37)
        f.update({'Target': owner.index, 'Interpolator': self.blend(kind).index})
        if extra:
            f.update(extra)
        c = self.b.add(ctype, **f)
        self._chain_attach(owner, c)
        self.ctrl_cache[key] = c
        return c

    # ---- main
    def run(self):
        b, root = self.b, self.root
        if not self.actions:
            return
        stop_all = max([a.stop for a in self.actions] + [1.0])
        mtc = b.add('NiMultiTargetTransformController', **dict(
            self._time_fields(FLT_MAX, -FLT_MAX, 44), Target=root.index, **{'Num Extra Targets': 0,
                                                                          'Extra Targets': np.zeros(0, np.int32)}))
        mgr = b.add('NiControllerManager', **dict(
            self._time_fields(FLT_MAX, -FLT_MAX, 12), Target=root.index, Cumulative=False,
            **{'Next Controller': mtc.index, 'Num Controller Sequences': 0,
               'Controller Sequences': np.zeros(0, np.int32), 'Object Palette': -1}))
        mgr['Next Controller'] = mtc.index
        self._chain_attach(root, mgr)
        seqs = []
        for ad in self.actions:
            seq = self.sequence(ad, mgr, mtc, stop_all)
            if seq is not None:
                seqs.append(seq)
        if not seqs:                                          # nothing usable: drop the manager again
            root['Controller'] = -1
            return
        mgr['Num Controller Sequences'] = len(seqs)
        mgr['Controller Sequences'] = np.array([s.index for s in seqs], dtype=np.int32)
        pal = b.add('NiStringPalette', Palette={'Palette': self.strings, 'Length': len(self.strings)})
        for s in seqs:
            s['String Palette'] = pal.index
            for cb in s['Controlled Blocks']:
                cb['String Palette'] = pal.index
        names = []
        seen = set()
        for blk in b.blocks:
            if blk.btype in _AV and str(blk.get('Name', '')) and str(blk['Name']) not in seen:
                seen.add(str(blk['Name']))
                names.append({'Name': str(blk['Name']), 'AV Object': blk.index})
        op = b.add('NiDefaultAVObjectPalette', Scene=root.index, **{'Num Objs': len(names), 'Objs': names})
        mgr['Object Palette'] = op.index
        extra = sorted({t.index for t in self.targets} | {root.index})
        mtc['Num Extra Targets'] = len(extra)
        mtc['Extra Targets'] = np.array(extra, dtype=np.int32)

    def sequence(self, ad, mgr, mtc, stop_all):
        b = self.b
        ctrl_blocks = []
        stop = max(ad.stop, max([_last_time(c) for c in ad.channels] + [0.0]))
        for ch in ad.channels:
            try:
                ctrl_blocks += self.channel_blocks(ch, ad, mtc, stop_all)
            except _Skip as e:
                self.warn.append("Animation '%s': %s" % (ad.name, e))
        if not ctrl_blocks:
            self.warn.append("Animation '%s' has no exportable channels; skipped." % ad.name)
            return None
        tk = None
        keys = list(ad.text_keys)
        if not any(v.lower() == 'start' for _, v in keys):
            keys.insert(0, (ad.start, 'start'))
        if not any(v.lower() == 'end' for _, v in keys):
            keys.append((stop, 'end'))
        keys.sort(key=lambda x: x[0])
        tk = b.add('NiTextKeyExtraData', Name='', **{'Num Text Keys': len(keys),
                                                      'Text Keys': [{'Time': float(t), 'Value': v} for t, v in keys]})
        names = {str(x['Name']) for x in b.blocks if x.btype in _AV}
        accum = ad.accum_root if ad.accum_root in names else str(self.root['Name'])
        seq = b.add('NiControllerSequence', Name=ad.name, **{
            'Num Controlled Blocks': len(ctrl_blocks), 'Array Grow By': 1, 'Controlled Blocks': ctrl_blocks,
            'Weight': float(ad.weight), 'Text Keys': tk.index, 'Cycle Type': CYCLE.get(ad.cycle, 0),
            'Frequency': float(ad.frequency or 1.0), 'Start Time': float(ad.start), 'Stop Time': float(stop),
            'Manager': mgr.index, 'Accum Root Name': accum, 'String Palette': -1})
        return seq

    def cblock(self, interp, ctrl, node_name, prop=None, ctype=None, cid=None):
        return {'Interpolator': interp.index, 'Controller': ctrl.index, 'Priority': 0, 'String Palette': -1,
                'Node Name Offset': self.s(node_name), 'Property Type Offset': self.s(prop),
                'Controller Type Offset': self.s(ctype), 'Controller ID Offset': self.s(cid),
                'Interpolator ID Offset': NO_OFFSET}

    def channel_blocks(self, ch, ad, mtc, stop_all):
        b = self.b
        tgt = ch.target
        if ch.kind in ('TRANSFORM', 'BONE'):
            if ch.kind == 'BONE':
                blk, tr = self.bone_tracks(tgt, ch.bone, ch)
                if blk is None:
                    raise _Skip("bone '%s' was not exported." % ch.bone)
                loc, quat, scale = tr
                euler = None
            else:
                blk = b.node_of.get(id(tgt)) or b.shape_of.get(id(tgt), (None,))[0]
                if blk is None:
                    raise _Skip("object '%s' was not exported." % tgt.name)
                loc, quat, euler, scale = ch.loc, ch.quat, ch.euler, ch.scale
            t0 = ad.start
            t1 = max(ad.stop, _last_time(ch))
            interp = self.transform_interp(blk, loc, quat, euler, scale, t0, t1)
            self.targets.append(blk)
            return [self.cblock(interp, mtc, str(blk['Name']), None, 'NiTransformController')]
        info = b.shape_of.get(id(tgt))
        if info is None:
            raise _Skip("object '%s' was not exported." % tgt.name)
        shape, mat = info
        if ch.kind == 'VISIBILITY':
            data = b.add('NiBoolData', Data=bool_keygroup(ch.values))
            interp = b.add('NiBoolInterpolator', Value=bool(ch.values.values[0][0] > 0.5), Data=data.index)
            ctrl = self.property_controller(shape, 'NiVisController', 'bool', ad.stop)
            return [self.cblock(interp, ctrl, str(shape['Name']), None, 'NiVisController')]
        if mat is None:
            raise _Skip("'%s' has no material to animate." % tgt.name)
        if ch.kind == 'MATERIAL_ALPHA':
            data = b.add('NiFloatData', Data=keygroup(ch.values, 0, 1))
            interp = b.add('NiFloatInterpolator', Value=float(ch.values.values[0][0]), Data=data.index)
            ctrl = self.property_controller(mat, 'NiAlphaController', 'float', ad.stop)
            return [self.cblock(interp, ctrl, str(shape['Name']), 'NiMaterialProperty', 'NiAlphaController')]
        code, cid = COLOR_SLOT.get(ch.slot, COLOR_SLOT['emissive'])
        data = b.add('NiPosData', Data=keygroup(ch.values, None, 3))
        interp = b.add('NiPoint3Interpolator', Value=_vec(ch.values.values[0]), Data=data.index)
        ctrl = self.property_controller(mat, 'NiMaterialColorController', 'point3', ad.stop, {'Target Color': code})
        return [self.cblock(interp, ctrl, str(shape['Name']), 'NiMaterialProperty', 'NiMaterialColorController', cid)]


class _Skip(Exception):
    pass


_AV = {'NiNode', 'NiTriShape', 'NiTriStrips', 'NiBillboardNode', 'NiSwitchNode', 'NiLODNode', 'BSFadeNode',
       'NiParticleSystem', 'NiCamera', 'NiLight'}


def _last_time(ch):
    t = 0.0
    for tr in [ch.loc, ch.quat, ch.scale, ch.values] + list(ch.euler or []):
        if tr is not None and len(tr.times):
            t = max(t, float(tr.times[-1]))
    return t


def _euler_of(R):
    """Rotation matrix -> XYZ euler angles (Blender convention R = Rz Ry Rx)."""
    import math
    sy = -R[2, 0]
    y = math.asin(max(-1.0, min(1.0, sy)))
    if abs(sy) < 0.999999:
        return [math.atan2(R[2, 1], R[2, 2]), y, math.atan2(R[1, 0], R[0, 0])]
    return [math.atan2(-R[1, 2], R[1, 1]), y, 0.0]
