# SPDX-License-Identifier: GPL-3.0-or-later
"""Creates Blender actions from :class:`nif_scene.ActionDesc` (Blender 4.0 API)."""
import numpy as np

import bpy


def _frames(times, fps):
    return np.asarray(times, dtype=np.float64) * fps


def add_fcurve(act, data_path, index, track, comp, fps, group=None, cycle='CLAMP', scale=1.0, negate=False):
    """One F-curve from component ``comp`` of a KeyTrack."""
    fc = act.fcurves.new(data_path, index=index, action_group=group) if group else act.fcurves.new(data_path, index=index)
    n = len(track.times)
    fr = _frames(track.times, fps)
    val = track.values[:, comp] * scale
    if negate:
        val = 1.0 - val
    kps = fc.keyframe_points
    kps.add(n)
    co = np.empty(n * 2)
    co[0::2], co[1::2] = fr, val
    kps.foreach_set('co', co)
    if track.interp == 'CONSTANT':
        for kp in kps:
            kp.interpolation = 'CONSTANT'
    elif track.interp == 'BEZIER' and track.fwd is not None:
        fwd = track.fwd[:, comp] * scale
        bwd = track.bwd[:, comp] * scale
        if negate:
            fwd, bwd = -fwd, -bwd
        for i, kp in enumerate(kps):
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = kp.handle_right_type = 'FREE'
            t = track.times[i]
            dt_next = (track.times[i + 1] - t) if i + 1 < n else ((t - track.times[i - 1]) if i > 0 else 0.0)
            dt_prev = (t - track.times[i - 1]) if i > 0 else dt_next
            kp.handle_left = (fr[i] - dt_prev * fps / 3.0, val[i] - bwd[i] * dt_prev / 3.0)
            kp.handle_right = (fr[i] + dt_next * fps / 3.0, val[i] + fwd[i] * dt_next / 3.0)
    elif track.interp == 'BEZIER':
        for kp in kps:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'
    else:
        for kp in kps:
            kp.interpolation = 'LINEAR'
    if cycle == 'LOOP' or cycle == 'REVERSE':
        mod = fc.modifiers.new('CYCLES')
        if cycle == 'REVERSE':
            mod.mode_before = mod.mode_after = 'MIRROR'
    fc.update()
    return fc


def _principled(mat):
    if mat is None or mat.node_tree is None:
        return None
    return next((n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled'), None)


def _socket_path(node, name):
    idx = list(node.inputs).index(node.inputs[name])
    return 'nodes["%s"].inputs[%d].default_value' % (node.name, idx)


def build_actions(desc, objects, context, settings):
    """Instantiate ``desc.actions``.  ``objects`` maps id(ObjDesc) -> bpy Object."""
    fps = float(settings.anim_fps)
    scene = context.scene
    warnings = []
    first = True
    for ad in desc.actions:
        blocks = {}                                   # id(datablock) -> (datablock, action)

        def action_for(block, label, owner=None):
            key = id(block)
            if key not in blocks:
                act = bpy.data.actions.new('%s_%s' % (ad.name, label))
                act.use_fake_user = True
                act['nif_sequence'] = ad.name
                act['nif_target'] = (owner or label)
                act['nif_cycle'] = ad.cycle
                act['nif_frequency'] = float(ad.frequency)
                act['nif_start'] = float(ad.start)
                act['nif_stop'] = float(ad.stop)
                act['nif_weight'] = float(ad.weight)
                act['nif_accum_root'] = ad.accum_root
                act['nif_text_keys'] = '\n'.join('%.6f|%s' % (t, s) for t, s in ad.text_keys)
                for t, s in ad.text_keys:
                    m = act.pose_markers.new(s)
                    m.frame = int(round(t * fps))
                blocks[key] = (block, act)
            return blocks[key][1]

        for ch in ad.channels:
            obj = objects.get(id(ch.target))
            if obj is None:
                continue
            cyc = ad.cycle
            if ch.kind == 'TRANSFORM':
                act = action_for(obj, obj.name, obj.name)
                if ch.euler:
                    obj.rotation_mode = 'XYZ'
                    for i, tr in enumerate(ch.euler):
                        if tr is not None:
                            add_fcurve(act, 'rotation_euler', i, tr, 0, fps, cycle=cyc)
                if ch.quat is not None:
                    obj.rotation_mode = 'QUATERNION'
                    for i in range(4):
                        add_fcurve(act, 'rotation_quaternion', i, ch.quat, i, fps, cycle=cyc)
                if ch.loc is not None:
                    for i in range(3):
                        add_fcurve(act, 'location', i, ch.loc, i, fps, cycle=cyc)
                if ch.scale is not None:
                    for i in range(3):
                        add_fcurve(act, 'scale', i, ch.scale, 0 if ch.scale.values.shape[1] == 1 else i, fps, cycle=cyc)
            elif ch.kind == 'BONE':
                act = action_for(obj, obj.name, obj.name)
                pb = obj.pose.bones.get(ch.bone) if obj.pose else None
                if pb is None:
                    continue
                pb.rotation_mode = 'QUATERNION'
                base = 'pose.bones["%s"].' % ch.bone
                for i in range(4):
                    add_fcurve(act, base + 'rotation_quaternion', i, ch.quat, i, fps, group=ch.bone, cycle=cyc)
                for i in range(3):
                    add_fcurve(act, base + 'location', i, ch.loc, i, fps, group=ch.bone, cycle=cyc)
                    add_fcurve(act, base + 'scale', i, ch.scale, i, fps, group=ch.bone, cycle=cyc)
            elif ch.kind == 'VISIBILITY':
                act = action_for(obj, obj.name, obj.name)
                for path in ('hide_viewport', 'hide_render'):
                    add_fcurve(act, path, 0, ch.values, 0, fps, cycle=cyc, negate=True)
            elif ch.kind in ('MATERIAL_COLOR', 'MATERIAL_ALPHA'):
                mat = obj.data.materials[0] if obj.type == 'MESH' and len(obj.data.materials) else None
                node = _principled(mat)
                if node is None:
                    warnings.append("No material to animate on '%s'." % obj.name)
                    continue
                act = action_for(mat.node_tree, mat.name, obj.name)
                if ch.kind == 'MATERIAL_ALPHA':
                    add_fcurve(act, _socket_path(node, 'Alpha'), 0, ch.values, 0, fps, cycle=cyc)
                    continue
                sock = {'emissive': 'Emission Color', 'diffuse': 'Base Color'}.get(ch.slot)
                if sock is None:
                    warnings.append("Animated %s colour of '%s' has no Blender equivalent; skipped." % (ch.slot, obj.name))
                    continue
                if sock == 'Emission Color':
                    node.inputs['Emission Strength'].default_value = 1.0
                path = _socket_path(node, sock)
                for i in range(3):
                    add_fcurve(act, path, i, ch.values, i, fps, cycle=cyc)
        for block, act in blocks.values():
            if block.animation_data is None:
                block.animation_data_create()
            if first or block.animation_data.action is None:
                block.animation_data.action = act
            act.use_fake_user = True
        if first:
            scene.render.fps = int(round(fps))
            scene.render.fps_base = 1.0
            scene.frame_start = int(round(ad.start * fps))
            scene.frame_end = max(scene.frame_start + 1, int(round(ad.stop * fps)))
            scene.frame_set(scene.frame_start)
            first = False
    return warnings
