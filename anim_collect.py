# SPDX-License-Identifier: GPL-3.0-or-later
"""Export side, step 1 (pure Python, no bpy): raw Blender F-curves -> :class:`nif_scene.ChannelDesc`.

``blender_export`` reads F-curves into :class:`RawCurve` objects; everything that converts them to key tracks
(Hermite tangents, quaternion densifying, euler orders, parent-inverse baking, bone sampling) lives here so it
can be unit-tested.  Times are seconds (``(frame - f0) / fps``); locations stay in Blender units (the NIF
writer divides by the export scale).
"""
import math
from dataclasses import dataclass

import numpy as np

from .nif_anim import euler_to_matrix, mat_to_quat, quat_to_mat, slerp
from .nif_scene import ChannelDesc, KeyTrack


@dataclass
class RawCurve:
    frames: np.ndarray            # (n,) key frames
    values: np.ndarray            # (n,)
    interp: list                  # per key: 'BEZIER' | 'LINEAR' | 'CONSTANT' (anything else is treated as LINEAR)
    hl: np.ndarray                # (n,2) left handle (frame, value)
    hr: np.ndarray                # (n,2) right handle (frame, value)


def eval_curve(c, frame):
    """Evaluate a RawCurve like Blender does (Bezier solved in x)."""
    f = c.frames
    if frame <= f[0]:
        return float(c.values[0])
    if frame >= f[-1]:
        return float(c.values[-1])
    i = int(np.searchsorted(f, frame, side='right')) - 1
    v0, v1 = float(c.values[i]), float(c.values[i + 1])
    mode = c.interp[i]
    if mode == 'CONSTANT':
        return v0
    x0, x1 = float(f[i]), float(f[i + 1])
    if mode != 'BEZIER':
        return v0 + (v1 - v0) * (frame - x0) / (x1 - x0)
    px = (x0, float(c.hr[i][0]), float(c.hl[i + 1][0]), x1)
    py = (v0, float(c.hr[i][1]), float(c.hl[i + 1][1]), v1)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        t = 0.5 * (lo + hi)
        u = 1 - t
        x = u ** 3 * px[0] + 3 * u * u * t * px[1] + 3 * u * t * t * px[2] + t ** 3 * px[3]
        if x < frame:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    u = 1 - t
    return u ** 3 * py[0] + 3 * u * u * t * py[1] + 3 * u * t * t * py[2] + t ** 3 * py[3]


def _union(curves):
    fr = np.unique(np.round(np.concatenate([c.frames for c in curves if c is not None]), 4))
    return fr


def _same_keys(curves, frames):
    return all(len(c.frames) == len(frames) and np.allclose(c.frames, frames, atol=1e-4) for c in curves if c is not None)


def _slope(y, y0, x, x0, fps, sign):
    dx = (x - x0) * sign / fps
    return 0.0 if abs(dx) < 1e-9 else (y - y0) * sign / dx


def track_from_curves(curves, base, fps, f0, warnings=None, label=''):
    """``curves``: list (one per component, ``None`` = not animated), ``base``: constant for missing components."""
    pres = [c for c in curves if c is not None]
    if not pres:
        return None
    frames = _union(curves)
    n, k = len(frames), len(curves)
    same = _same_keys(curves, frames)
    vals = np.empty((n, k))
    for j, c in enumerate(curves):
        if c is None:
            vals[:, j] = base[j]
        elif same:
            vals[:, j] = c.values
        else:
            vals[:, j] = [eval_curve(c, x) for x in frames]
    times = (frames - f0) / fps
    modes = []
    if same:
        for i in range(max(n - 1, 1)):
            modes.append({(c.interp[i] if c.interp[i] in ('BEZIER', 'CONSTANT') else 'LINEAR') for c in pres})
    flat = {m for s in modes for m in s}
    if not same:
        mode = 'CONSTANT' if all(all(m == 'CONSTANT' for m in c.interp[:-1]) for c in pres) else 'LINEAR'
    elif len(flat) == 1:
        mode = next(iter(flat))
    else:
        cnt = {m: sum(1 for s in modes for x in s if x == m) for m in flat}
        mode = max(cnt, key=cnt.get)
        if warnings is not None:
            warnings.append("%s: mixed key interpolation modes; using %s for the whole curve." % (label, mode))
    fwd = bwd = None
    if mode == 'BEZIER':
        fwd, bwd = np.zeros((n, k)), np.zeros((n, k))
        for j, c in enumerate(curves):
            if c is None:
                continue
            for i in range(n):
                x, y = float(c.frames[i]), float(c.values[i])
                fwd[i, j] = _slope(c.hr[i][1], y, c.hr[i][0], x, fps, 1)
                bwd[i, j] = _slope(y, c.hl[i][1], x, c.hl[i][0], fps, 1)
    if n == 1 and mode == 'BEZIER':
        mode, fwd, bwd = 'LINEAR', None, None
    return KeyTrack(times, vals, mode, fwd, bwd)


def _quat_frames(curves, frames_union):
    """Union key frames, densified to one sample per frame wherever a Bezier segment is longer than a frame."""
    pres = [c for c in curves if c is not None]
    fr = list(frames_union)
    extra = []
    bezier = any('BEZIER' in c.interp for c in pres)
    for i in range(len(fr) - 1):
        if bezier and fr[i + 1] - fr[i] > 1.5:
            extra += list(np.arange(np.floor(fr[i]) + 1, fr[i + 1] - 1e-6))
    return np.unique(np.round(np.concatenate([frames_union, np.array(extra, dtype=np.float64)]), 4))


def quat_track_from_curves(curves, base, fps, f0):
    pres = [c for c in curves if c is not None]
    if not pres:
        return None
    frames = _quat_frames(curves, _union(curves))
    q = np.empty((len(frames), 4))
    for j, c in enumerate(curves):
        q[:, j] = base[j] if c is None else [eval_curve(c, x) for x in frames]
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(nrm < 1e-12, 1.0, nrm)
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    const = all(all(m == 'CONSTANT' for m in c.interp[:-1]) for c in pres)
    return KeyTrack((frames - f0) / fps, q, 'CONSTANT' if const else 'LINEAR')


def euler_matrix(e, order='XYZ'):
    ax = {'X': lambda a: euler_to_matrix(a, 0, 0), 'Y': lambda a: euler_to_matrix(0, a, 0),
          'Z': lambda a: euler_to_matrix(0, 0, a)}
    R = np.eye(3)
    for ch in order:                                   # first letter is applied first
        R = ax[ch]({'X': e[0], 'Y': e[1], 'Z': e[2]}[ch]) @ R
    return R


# ------------------------------------------------------------- object channels
def _get(curves, path, n):
    d = curves.get(path, {})
    return [d.get(i) for i in range(n)]


def _identity(m):
    return m is None or np.allclose(np.asarray(m), np.eye(4), atol=1e-6)


def _frame_range(curve_lists):
    allc = [c for cl in curve_lists for c in cl if c is not None]
    return min(c.frames[0] for c in allc), max(c.frames[-1] for c in allc)


def object_channel(target, curves, rest, fps, f0, rest_scale=None, pinv=None, warnings=None, label=''):
    """Blender object F-curves -> TRANSFORM channel.

    ``curves``  {data_path: {index: RawCurve}} for location / rotation_euler / rotation_quaternion / scale
    ``rest``    {'location','rotation_euler','rotation_quaternion','scale','rotation_mode'} current property values
    ``rest_scale``  scale baked into mesh vertices (3-vector) or ``None`` for nodes (absolute scale is exported)
    ``pinv``    matrix that maps the local basis matrix to the exported parent-relative matrix
    """
    W = warnings if warnings is not None else []
    loc_c = _get(curves, 'location', 3)
    eul_c = _get(curves, 'rotation_euler', 3)
    quat_c = _get(curves, 'rotation_quaternion', 4)
    scl_c = _get(curves, 'scale', 3)
    mode = rest.get('rotation_mode', 'XYZ')
    if mode == 'QUATERNION':
        eul_c = [None] * 3
    else:
        quat_c = [None] * 4
    if mode == 'AXIS_ANGLE':
        W.append("%s: axis-angle rotation is not supported; rotation skipped." % label)
    if all(c is None for c in loc_c + eul_c + quat_c + scl_c):
        return None
    if not _identity(pinv):
        return _sampled_channel(target, loc_c, eul_c, quat_c, scl_c, rest, fps, f0, rest_scale, pinv)
    ch = ChannelDesc('TRANSFORM', target)
    ch.loc = track_from_curves(loc_c, rest['location'], fps, f0, W, label + ' location')
    if mode == 'QUATERNION':
        ch.quat = quat_track_from_curves(quat_c, rest['rotation_quaternion'], fps, f0)
    elif any(c is not None for c in eul_c):
        if mode == 'XYZ':
            ch.euler = []
            for i in range(3):
                ch.euler.append(track_from_curves([eul_c[i]], [rest['rotation_euler'][i]], fps, f0, W, label + ' rotation')
                                if eul_c[i] is not None else None)
        else:                                          # other euler orders: sample every frame as quaternions
            ch.quat = _sample_euler_quat(eul_c, rest['rotation_euler'], mode, fps, f0)
    st = track_from_curves(scl_c, rest['scale'], fps, f0, W, label + ' scale')
    if st is not None:
        ref = np.ones(3) if rest_scale is None else np.where(np.abs(rest_scale) < 1e-12, 1.0, rest_scale)
        ch.scale = KeyTrack(st.times, (st.values / ref).mean(axis=1, keepdims=True), st.interp,
                            None if st.fwd is None else (st.fwd / ref).mean(axis=1, keepdims=True),
                            None if st.bwd is None else (st.bwd / ref).mean(axis=1, keepdims=True))
    return ch


def _sample_euler_quat(eul_c, base, mode, fps, f0):
    lo, hi = _frame_range([eul_c])
    frames = np.unique(np.concatenate([np.arange(math.floor(lo), math.ceil(hi) + 1), _union(eul_c)]))
    frames = frames[(frames >= lo - 1e-6) & (frames <= hi + 1e-6)]
    q = np.empty((len(frames), 4))
    for i, x in enumerate(frames):
        e = [eval_curve(c, x) if c is not None else base[j] for j, c in enumerate(eul_c)]
        q[i] = mat_to_quat(euler_matrix(e, mode))
        if i and q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return KeyTrack((frames - f0) / fps, q, 'LINEAR')


def _sampled_channel(target, loc_c, eul_c, quat_c, scl_c, rest, fps, f0, rest_scale, pinv):
    lo, hi = _frame_range([loc_c, eul_c, quat_c, scl_c])
    frames = np.arange(math.floor(lo), math.ceil(hi) + 1.0)
    n = len(frames)
    L, Q, S = np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, 1))
    mode = rest.get('rotation_mode', 'XYZ')
    ref = np.ones(3) if rest_scale is None else np.where(np.abs(rest_scale) < 1e-12, 1.0, rest_scale)
    for i, x in enumerate(frames):
        loc = [eval_curve(c, x) if c is not None else rest['location'][j] for j, c in enumerate(loc_c)]
        scl = [eval_curve(c, x) if c is not None else rest['scale'][j] for j, c in enumerate(scl_c)]
        if mode == 'QUATERNION':
            q = np.array([eval_curve(c, x) if c is not None else rest['rotation_quaternion'][j] for j, c in enumerate(quat_c)])
            R = quat_to_mat(q / (np.linalg.norm(q) or 1.0))
        else:
            e = [eval_curve(c, x) if c is not None else rest['rotation_euler'][j] for j, c in enumerate(eul_c)]
            R = euler_matrix(e, mode if mode != 'AXIS_ANGLE' else 'XYZ')
        B = np.eye(4)
        B[:3, :3] = R * np.array(scl)
        B[:3, 3] = loc
        M = np.asarray(pinv) @ B
        cols = np.linalg.norm(M[:3, :3], axis=0)
        L[i] = M[:3, 3]
        q = mat_to_quat(M[:3, :3] / np.where(cols == 0, 1, cols))
        if i and q @ Q[i - 1] < 0:
            q = -q
        Q[i] = q
        S[i, 0] = float((cols / ref).mean()) if rest_scale is not None else float(cols.mean())
    t = (frames - f0) / fps
    ch = ChannelDesc('TRANSFORM', target, loc=KeyTrack(t, L), quat=KeyTrack(t, Q))
    if scl_c is not None and (any(c is not None for c in scl_c) or not _identity(pinv)):
        ch.scale = KeyTrack(t, S)
    return ch


# ---------------------------------------------------------------------- bones
def bone_channel(arm, bone, curves, rest, fps, f0, warnings=None, label=''):
    """Pose-bone F-curves -> BONE channel sampled once per frame (pose basis in bone space, as the importer writes)."""
    base = 'pose.bones["%s"].' % bone
    loc_c = _get(curves, base + 'location', 3)
    quat_c = _get(curves, base + 'rotation_quaternion', 4)
    eul_c = _get(curves, base + 'rotation_euler', 3)
    scl_c = _get(curves, base + 'scale', 3)
    mode = rest.get('rotation_mode', 'QUATERNION')
    if mode == 'QUATERNION':
        eul_c = [None] * 3
    else:
        quat_c = [None] * 4
    if all(c is None for c in loc_c + quat_c + eul_c + scl_c):
        return None
    lo, hi = _frame_range([loc_c, quat_c, eul_c, scl_c])
    frames = np.unique(np.concatenate([np.arange(math.floor(lo), math.ceil(hi) + 1.0), _union(loc_c + quat_c + eul_c + scl_c)]))
    frames = frames[(frames >= lo - 1e-6) & (frames <= hi + 1e-6)]
    n = len(frames)
    L, Q, S = np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, 3))
    for i, x in enumerate(frames):
        L[i] = [eval_curve(c, x) if c is not None else rest['location'][j] for j, c in enumerate(loc_c)]
        S[i] = [eval_curve(c, x) if c is not None else rest['scale'][j] for j, c in enumerate(scl_c)]
        if mode == 'QUATERNION':
            q = np.array([eval_curve(c, x) if c is not None else rest['rotation_quaternion'][j] for j, c in enumerate(quat_c)])
            q = q / (np.linalg.norm(q) or 1.0)
        else:
            e = [eval_curve(c, x) if c is not None else rest['rotation_euler'][j] for j, c in enumerate(eul_c)]
            q = mat_to_quat(euler_matrix(e, mode if mode != 'AXIS_ANGLE' else 'XYZ'))
        if i and q @ Q[i - 1] < 0:
            q = -q
        Q[i] = q
    t = (frames - f0) / fps
    return ChannelDesc('BONE', arm, bone=bone, loc=KeyTrack(t, L), quat=KeyTrack(t, Q), scale=KeyTrack(t, S))


# ---------------------------------------------------------- material / visibility
def values_channel(kind, target, curve_list, base, fps, f0, slot='', warnings=None, label='', invert=False):
    tr = track_from_curves(curve_list, base, fps, f0, warnings, label)
    if tr is None:
        return None
    if invert:
        tr = KeyTrack(tr.times, 1.0 - tr.values, tr.interp, None if tr.fwd is None else -tr.fwd,
                      None if tr.bwd is None else -tr.bwd)
    if kind == 'VISIBILITY':
        tr = KeyTrack(tr.times, (tr.values > 0.5).astype(np.float64), 'CONSTANT')
    return ChannelDesc(kind, target, values=tr, slot=slot)
