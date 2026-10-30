# SPDX-License-Identifier: GPL-3.0-or-later
"""Keyframe animation import (pure Python): NiControllerManager sequences and plain node controllers -> ActionDesc.

Supported
  * NiTransformInterpolator / NiKeyframeData (translation, quaternion or XYZ-euler rotation, scale)
  * NiMaterialColorController (ambient / diffuse / specular / self-illumination colour)
  * NiAlphaController (material alpha), NiVisController (visibility)
  * text keys (markers), cycle type, frequency, weight, accum root (stored as custom properties)
Not supported (a warning is reported): particle-system controllers, UV/texture transforms, flip, morph, B-spline data.
"""
import math

import numpy as np

from .nif_scene import ActionDesc, ChannelDesc, KeyTrack, bone_axis_matrix, scale_translation, transform_matrix

FLT_INVALID = -3.0e38            # NiQuatTransform uses -FLT_MAX for "not animated"
CYCLES = {0: 'LOOP', 1: 'REVERSE', 2: 'CLAMP'}
COLOR_SLOTS = {0: 'ambient', 1: 'diffuse', 2: 'specular', 3: 'emissive'}
COLOR_IDS = {'AMBIENT': 'ambient', 'DIFFUSE': 'diffuse', 'SPECULAR': 'specular', 'SELF_ILLUM': 'emissive'}
KEY_LINEAR, KEY_QUADRATIC, KEY_TBC, KEY_XYZ, KEY_CONST = 1, 2, 3, 4, 5


# ------------------------------------------------------------------ key readers
def _val(v, k):
    if k == 1:
        return [float(v)]
    if hasattr(v, 'items'):
        if 'w' in v and k == 4:
            return [v['w'], v['x'], v['y'], v['z']]
        if 'r' in v:
            return [v['r'], v['g'], v['b']][:k]
        return [v['x'], v['y'], v['z']][:k]
    return list(v)


def keygroup_track(kg, k):
    """KeyGroup struct -> KeyTrack (or None).  ``k`` = components per value."""
    if kg is None or int(kg['Num Keys']) == 0:
        return None
    keys = kg['Keys']
    it = int(kg['Interpolation'])
    times = np.array([float(x['Time']) for x in keys], dtype=np.float64)
    vals = np.array([_val(x['Value'], k) for x in keys], dtype=np.float64).reshape(len(keys), k)
    interp, fwd, bwd = 'LINEAR', None, None
    if it == KEY_CONST:
        interp = 'CONSTANT'
    elif it == KEY_QUADRATIC:
        interp = 'BEZIER'
        fwd = np.array([_val(x['Forward'], k) for x in keys], dtype=np.float64).reshape(len(keys), k)
        bwd = np.array([_val(x['Backward'], k) for x in keys], dtype=np.float64).reshape(len(keys), k)
    elif it == KEY_TBC:
        interp = 'BEZIER'                                  # approximated with automatic handles (fwd = None)
    return KeyTrack(times, vals, interp, fwd, bwd)


def quat_track(data):
    n = int(data['Num Rotation Keys'])
    if n == 0 or int(data['Rotation Type']) == KEY_XYZ:
        return None
    keys = data['Quaternion Keys']
    times = np.array([float(x['Time']) for x in keys], dtype=np.float64)
    vals = np.array([_val(x['Value'], 4) for x in keys], dtype=np.float64).reshape(n, 4)
    nrm = np.linalg.norm(vals, axis=1, keepdims=True)
    vals = vals / np.where(nrm == 0, 1, nrm)
    for i in range(1, n):                                  # keep neighbouring quaternions in the same hemisphere
        if vals[i] @ vals[i - 1] < 0:
            vals[i] = -vals[i]
    return KeyTrack(times, vals, 'CONSTANT' if int(data['Rotation Type']) == KEY_CONST else 'LINEAR')


def bool_track(kg):
    t = keygroup_track(kg, 1)
    if t is not None:
        t.interp = 'CONSTANT'
        t.fwd = t.bwd = None
    return t


def vis_track(data):
    keys = data['Keys']
    if len(keys) == 0:
        return None
    return KeyTrack(np.array([float(x['Time']) for x in keys]), np.array([[float(x['Value'])] for x in keys]), 'CONSTANT')


# ---------------------------------------------------------------- evaluation
def slerp(q0, q1, u):
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / (np.linalg.norm(q) or 1.0)


def eval_track(tr, t, quat=False):
    times = tr.times
    if len(times) == 1 or t <= times[0]:
        return tr.values[0].copy()
    if t >= times[-1]:
        return tr.values[-1].copy()
    i = int(np.searchsorted(times, t, side='right')) - 1
    t0, t1 = times[i], times[i + 1]
    u = (t - t0) / (t1 - t0)
    v0, v1 = tr.values[i], tr.values[i + 1]
    if tr.interp == 'CONSTANT':
        return v0.copy()
    if quat:
        return slerp(v0, v1, u)
    if tr.interp == 'BEZIER' and tr.fwd is not None:
        dt = t1 - t0
        h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        return h00 * v0 + h10 * dt * tr.fwd[i] + h01 * v1 + h11 * dt * tr.bwd[i + 1]
    return v0 + u * (v1 - v0)


def euler_to_matrix(x, y, z):
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                                    # same convention as Blender's 'XYZ' euler


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / (np.linalg.norm(q) or 1.0)


def _valid(v):
    return all(float(x) > FLT_INVALID for x in v)


# ------------------------------------------------------------------- importer
class AnimImporter:
    def __init__(self, sb):
        self.sb = sb                           # SceneBuilder
        self.nif = sb.nif
        self.warned = set()
        self.by_name = {}
        for b in self.nif.blocks:
            if b.is_a('NiAVObject'):
                self.by_name.setdefault(str(b['Name']), []).append(b)
        self.bone_owner = {}
        for arm in sb.armatures.values():
            for bd in arm.bones:
                self.bone_owner[bd.name] = (arm, bd)

    def warn_once(self, msg):
        if msg not in self.warned:
            self.warned.add(msg)
            self.sb.warn.append(msg)

    # -- entry
    def run(self):
        sb = self.sb
        managers = [b for b in self.nif.blocks if b.btype == 'NiControllerManager']
        for m in managers:
            for ref in m['Controller Sequences']:
                seq = sb.d(ref)
                if seq is not None:
                    act = self.sequence(seq)
                    if act is not None:
                        sb.scene.actions.append(act)
        if not managers:
            act = self.plain_controllers()
            if act is not None:
                sb.scene.actions.append(act)
        sb.scene.stats['actions'] = len(sb.scene.actions)

    # -- sequences
    def sequence(self, seq):
        sb = self.sb
        pal = sb.d(seq['String Palette'])
        text = pal['Palette']['Palette'] if pal is not None else ''

        def st(off):
            if off is None or int(off) == 0xFFFFFFFF:
                return None
            off = int(off)
            end = text.find('\x00', off)
            return text[off:end if end >= 0 else None]

        act = ActionDesc(str(seq['Name']) or 'Sequence', start=float(seq['Start Time']), stop=float(seq['Stop Time']),
                         frequency=float(seq['Frequency']), cycle=CYCLES.get(int(seq['Cycle Type']), 'LOOP'),
                         weight=float(seq['Weight']), accum_root=str(seq['Accum Root Name'] or ''))
        tk = sb.d(seq['Text Keys'])
        if tk is not None:
            act.text_keys = [(float(k['Time']), str(k['Value'])) for k in tk['Text Keys']]
        for cb in seq['Controlled Blocks']:
            interp = sb.d(cb['Interpolator'])
            if interp is None:
                continue
            ctrl = sb.d(cb['Controller'])
            node = st(cb['Node Name Offset']) if pal is not None else str(cb['Node Name'])
            prop = st(cb['Property Type Offset']) if pal is not None else str(cb['Property Type'])
            ctype = st(cb['Controller Type Offset']) if pal is not None else str(cb['Controller Type'])
            cid = st(cb['Controller ID Offset']) if pal is not None else str(cb['Controller ID'])
            self.channel(act, node, prop, ctype, cid, ctrl, interp)
        return act

    def plain_controllers(self):
        sb = self.sb
        act = ActionDesc('Default')
        found = False
        for b in self.nif.blocks:
            targets = []
            if b.is_a('NiAVObject') and int(b['Controller']) >= 0:
                targets.append((b, None, str(b['Name'])))
            if b.btype == 'NiMaterialProperty' and int(b['Controller']) >= 0:
                targets.append((b, 'NiMaterialProperty', None))
            for blk, prop, name in targets:
                c = sb.d(blk['Controller'])
                while c is not None:
                    interp = sb.d(c['Interpolator']) if 'Interpolator' in c else None
                    data = sb.d(c['Data']) if 'Data' in c else None
                    if c.btype in ('NiTransformController', 'NiKeyframeController', 'NiMaterialColorController',
                                   'NiAlphaController', 'NiVisController'):
                        nodes = [name]
                        if prop is not None:                       # property controller: find the shapes using it
                            nodes = [str(s['Name']) for s in self.nif.blocks if s.is_a('NiAVObject') and
                                     any(int(p) == blk.index for p in s['Properties'])]
                        for nd in nodes:
                            if self.channel(act, nd, prop, c.btype, None, c, interp, data):
                                st_, sp_ = float(c['Start Time']), float(c['Stop Time'])
                                if sp_ > 1e30 or st_ < -1e30:        # unset (FLT_MAX / -FLT_MAX)
                                    st_, sp_ = 0.0, 0.0
                                act.start = st_ if not found else min(act.start, st_)
                                act.stop = max(act.stop, sp_)
                                act.frequency = float(c['Frequency']) or 1.0
                                act.cycle = CYCLES.get((int(c['Flags']) >> 1) & 3, 'LOOP')
                                found = True
                    elif c.btype not in ('NiControllerManager', 'NiMultiTargetTransformController'):
                        self.warn_once('Controller %s is not supported and was skipped.' % c.btype)
                    c = sb.d(c['Next Controller'])
        return act if found else None

    # -- channels
    def find_target(self, name, want_geom=False):
        """name -> ('OBJ', ObjDesc) | ('BONE', ArmatureObjDesc, BoneDesc, nif block) | None"""
        sb = self.sb
        for blk in self.by_name.get(name, []):
            o = sb.obj_of.get(blk.index)
            if o is not None and o.kind != 'ARMATURE':
                return ('OBJ', o, blk)
        if name in self.bone_owner:
            arm, bd = self.bone_owner[name]
            blk = next((b for b in self.by_name.get(name, [])), None)
            return ('BONE', arm, bd, blk)
        for blk in self.by_name.get(name, []):                      # armature root node
            o = sb.obj_of.get(blk.index)
            if o is not None:
                return ('OBJ', o, blk)
        return None

    def channel(self, act, node, prop, ctype, cid, ctrl, interp, data=None):
        sb = self.sb
        ctype = ctype or (ctrl.btype if ctrl is not None else '')
        if ctype in ('NiTransformController', 'NiKeyframeController'):
            return self.transform_channel(act, node, interp, data)
        if ctype == 'NiMaterialColorController':
            tgt = self.find_target(node)
            if tgt is None or tgt[0] != 'OBJ':
                return False
            slot = COLOR_IDS.get(str(cid)) if cid else None
            if slot is None and ctrl is not None and 'Target Color' in ctrl:
                slot = COLOR_SLOTS.get(int(ctrl['Target Color']), 'emissive')
            d = data if data is not None else sb.d(interp['Data']) if interp is not None and 'Data' in interp else None
            tr = keygroup_track(d['Data'], 3) if d is not None else None
            if tr is None:
                return False
            act.channels.append(ChannelDesc('MATERIAL_COLOR', tgt[1], values=tr, slot=slot or 'emissive'))
            return True
        if ctype == 'NiAlphaController':
            tgt = self.find_target(node)
            d = data if data is not None else sb.d(interp['Data']) if interp is not None and 'Data' in interp else None
            tr = keygroup_track(d['Data'], 1) if d is not None else None
            if tgt is None or tgt[0] != 'OBJ' or tr is None:
                return False
            act.channels.append(ChannelDesc('MATERIAL_ALPHA', tgt[1], values=tr))
            return True
        if ctype == 'NiVisController':
            tgt = self.find_target(node)
            d = data if data is not None else sb.d(interp['Data']) if interp is not None and 'Data' in interp else None
            if tgt is None or tgt[0] != 'OBJ' or d is None:
                return False
            tr = vis_track(d) if d.btype == 'NiVisData' else bool_track(d['Data'])
            if tr is None:
                return False
            act.channels.append(ChannelDesc('VISIBILITY', tgt[1], values=tr))
            return True
        self.warn_once('Animation controller %s (e.g. particles, UV scroll, morphs) is not supported and was skipped.' % ctype)
        return False

    def transform_channel(self, act, node, interp, data):
        sb = self.sb
        tgt = self.find_target(node)
        if tgt is None:
            self.warn_once("Animated node '%s' was not imported; its animation was skipped." % node)
            return False
        if interp is not None and interp.btype == 'NiTransformInterpolator':
            data = sb.d(interp['Data'])
            base = interp['Transform']
        elif interp is not None and interp.is_a('NiBSplineInterpolator'):
            self.warn_once('B-spline compressed animation is not supported yet.')
            return False
        else:
            base = None
        blk = tgt[2] if tgt[0] == 'OBJ' else tgt[3]
        # node rest (NIF local) for constant channels
        rest_T = rest_q = None
        s = sb.s.scale
        loc = quat = scl = None
        eul = None
        if data is not None:
            kg = data['Translations']
            loc = keygroup_track(kg, 3)
            if int(data['Num Rotation Keys']) > 0:
                if int(data['Rotation Type']) == KEY_XYZ:
                    eul = [keygroup_track(g, 1) for g in data['XYZ Rotations']]
                    if all(e is None for e in eul):
                        eul = None
                else:
                    quat = quat_track(data)
            scl = keygroup_track(data['Scales'], 1)
        if base is not None and blk is not None:
            rest = transform_matrix(blk)
            if loc is None and _valid(_xyz(base['Translation'])):
                v = np.array(_xyz(base['Translation']))
                if np.abs(v - rest[:3, 3]).max() > 1e-4:
                    loc = KeyTrack(np.array([act.start]), v.reshape(1, 3))
            if quat is None and eul is None:
                q = base['Rotation']
                qq = np.array([q['w'], q['x'], q['y'], q['z']], dtype=np.float64)
                if _valid(qq):
                    Rr = rest[:3, :3] / (np.cbrt(abs(np.linalg.det(rest[:3, :3]))) or 1.0)
                    if np.abs(quat_to_mat(qq / (np.linalg.norm(qq) or 1)) - Rr).max() > 1e-4:
                        quat = KeyTrack(np.array([act.start]), (qq / (np.linalg.norm(qq) or 1)).reshape(1, 4))
            if scl is None and float(base['Scale']) > FLT_INVALID:
                sc = float(np.cbrt(abs(np.linalg.det(rest[:3, :3]))) or 1.0)
                if abs(float(base['Scale']) - sc) > 1e-4:
                    scl = KeyTrack(np.array([act.start]), np.array([[float(base['Scale'])]]))
        if loc is None and quat is None and eul is None and scl is None:
            return False
        if tgt[0] == 'OBJ':
            if loc is not None:
                loc = KeyTrack(loc.times, loc.values * s, loc.interp,
                               None if loc.fwd is None else loc.fwd * s, None if loc.bwd is None else loc.bwd * s)
            act.channels.append(ChannelDesc('TRANSFORM', tgt[1], loc=loc, quat=quat, euler=eul, scale=scl))
        else:
            ch = self.bake_bone(tgt[1], tgt[2], loc, quat, eul, scl, act)
            if ch is not None:
                act.channels.append(ch)
        return True

    # -- bones: sample at the frame rate in bone space
    def bake_bone(self, arm, bd, loc, quat, eul, scl, act):
        sb = self.sb
        fps = sb.s.anim_fps
        s = sb.s.scale
        C = np.eye(4)
        C[:3, :3] = bone_axis_matrix(int(arm.props.get('nif_bone_forward', 0)))
        parent = next((b for b in arm.bones if b.name == bd.parent), None)
        rest = np.asarray(bd.nif_matrix, dtype=np.float64)
        if parent is not None:
            rest = np.linalg.inv(np.asarray(parent.nif_matrix, dtype=np.float64)) @ rest
        rest_b = scale_translation(rest, s)
        rest_inv = np.linalg.inv(rest_b)
        rs = float(np.cbrt(abs(np.linalg.det(rest[:3, :3]))) or 1.0)
        rest_R = rest[:3, :3] / rs
        rest_q = mat_to_quat(rest_R)
        tracks = [t for t in ([loc, quat, scl] + list(eul or [])) if t is not None]
        t0 = min(float(t.times[0]) for t in tracks)
        t1 = max(float(t.times[-1]) for t in tracks)
        n0, n1 = int(math.ceil(t0 * fps - 1e-6)), int(math.floor(t1 * fps + 1e-6))
        times = sorted({t0, t1, *[k / fps for k in range(n0, n1 + 1)]})
        L = np.zeros((len(times), 3))
        Q = np.zeros((len(times), 4))
        S = np.zeros((len(times), 3))
        prev = None
        Cinv = np.linalg.inv(C)
        for i, t in enumerate(times):
            T = eval_track(loc, t) if loc is not None else rest[:3, 3]
            if quat is not None:
                R = quat_to_mat(eval_track(quat, t, quat=True))
            elif eul is not None:
                e = [float(eval_track(tr, t)[0]) if tr is not None else 0.0 for tr in eul]
                R = euler_to_matrix(*e)
            else:
                R = rest_R
            sc = float(eval_track(scl, t)[0]) if scl is not None else rs
            Lt = np.eye(4)
            Lt[:3, :3] = R * sc
            Lt[:3, 3] = T * s
            P = Cinv @ rest_inv @ Lt @ C
            L[i] = P[:3, 3]
            cols = P[:3, :3]
            sv = np.linalg.norm(cols, axis=0)
            S[i] = sv
            q = mat_to_quat(cols / np.where(sv == 0, 1, sv))
            if prev is not None and q @ prev < 0:
                q = -q
            Q[i] = prev = q
        tt = np.array(times)
        return ChannelDesc('BONE', arm, bone=bd.name, loc=KeyTrack(tt, L), quat=KeyTrack(tt, Q), scale=KeyTrack(tt, S))


def _xyz(v):
    return [v['x'], v['y'], v['z']]
