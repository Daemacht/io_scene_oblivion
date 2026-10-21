# SPDX-License-Identifier: GPL-3.0-or-later
"""Havok collision export for Oblivion (bhkCollisionObject -> bhkRigidBody -> shape).

Layouts and constants follow the 2.49 NifTools exporter and vanilla meshes.

Supported shapes
  * BOX / SPHERE  -> bhkBoxShape / bhkSphereShape inside a bhkConvexTransformShape (as the 2.49 exporter did)
  * CAPSULE       -> bhkCapsuleShape
  * CONVEX        -> bhkConvexVerticesShape (vertices + planes)
  * MESH          -> bhkMoppBvTreeShape + bhkPackedNiTriStripsShape when the mesh is an UNCHANGED import
                     (the original MOPP code is kept in the ``nif_mopp_*`` properties);
                     otherwise bhkNiTriStripsShape.  A MOPP tree cannot be generated here (it needs Havok's
                     proprietary compiler), so edited / new meshes use the slower but valid strips shape.
"""
import base64
import hashlib

import numpy as np

HAVOK_SCALE = 7.0

# byte patterns copied from vanilla / the 2.49 exporter (all "unused" in nif.xml)
BOX_UNUSED = bytes([0x6b, 0xee, 0x43, 0x40, 0x3a, 0xef, 0x8e, 0x3e])
CONVEX_TF_UNUSED = bytes([96, 120, 53, 19, 24, 9, 253, 4])
MOPP_UNUSED = bytes.fromhex('00000000a00d4b01c0cf900b')
RB_WORLD_UNUSED2 = b'\xa1\x37\x7c'
RB_UNUSED01 = b';\x8c\xd0?'
RB_UNUSED02 = b' S\xe5\x11'
RB_UNUSED04 = b',\x00\x00\x00'
RB_UNUSED05 = b'\x00\x0c\xba\xbf%\xf2\xae\xc1\xfb\xb1\xf6\x04'

EMPTY_PROP = {'Data': 0, 'Size': 0, 'Capacity and Flags': 0}


def _v3(v):
    return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}


def _v4(v, w=0.0):
    return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2]), 'w': float(w)}


def hk_matrix44(M):
    """4x4 column-vector matrix (translation in column 3, Havok units) -> Matrix44 fields (m14/m24/m34 = translation)."""
    return {'m%d%d' % (i + 1, j + 1): float(M[i][j]) for i in range(4) for j in range(4)}


def matrix_to_quat(R):
    """3x3 rotation -> (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)
    q = np.array(q, dtype=np.float64)
    return q / (np.linalg.norm(q) or 1.0)


# ------------------------------------------------------------------ geometry
def fit_box(v):
    """Oriented box of a point set -> (center, R (columns = axes), half extents)."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 8:                                   # imported / modelled box: take the frame from the edges
        d = np.linalg.norm(v[:, None] - v[None], axis=2)
        nb = np.argsort(d[0])[1:4]
        e = [v[i] - v[0] for i in nb]
        ln = [np.linalg.norm(x) for x in e]
        if min(ln) > 1e-9 and all(abs(e[i] @ e[j]) < 1e-3 * ln[i] * ln[j] for i in range(3) for j in range(i + 1, 3)):
            R = np.stack([x / n for x, n in zip(e, ln)], axis=1)
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            c = v.mean(0)
            half = np.abs((v - c) @ R).max(0)
            return c, R, half
    return (v.min(0) + v.max(0)) / 2, np.eye(3), (v.max(0) - v.min(0)) / 2


def fit_sphere(v):
    v = np.asarray(v, dtype=np.float64)
    c = (v.min(0) + v.max(0)) / 2
    return c, float(np.linalg.norm(v - c, axis=1).max())


def fit_capsule(v):
    """Capsule through a point cloud -> (p1, p2, radius); None if it degenerates to a sphere."""
    v = np.asarray(v, dtype=np.float64)
    c0 = v.mean(0)
    _, _, vt = np.linalg.svd(v - c0, full_matrices=False)
    axis = vt[0]
    t = (v - c0) @ axis
    perp = (v - c0) - t[:, None] * axis
    r = float(np.linalg.norm(perp, axis=1).max())
    lo, hi = float(t.min()) + r, float(t.max()) - r
    if hi - lo < 1e-4:
        return None
    return c0 + axis * lo, c0 + axis * hi, r


def convex_hull(points, eps=1e-9):
    """Incremental 3D convex hull.  Returns (hull vertex array, faces (k,3) oriented outward as index into that array)."""
    P = np.unique(np.round(np.asarray(points, dtype=np.float64), 6), axis=0)
    n = len(P)
    if n < 4:
        raise ValueError('convex hull needs at least 4 distinct points')
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(P - P[i0], axis=1)))
    d = P[i1] - P[i0]
    i2 = int(np.argmax(np.linalg.norm(np.cross(P - P[i0], d), axis=1)))
    nrm = np.cross(P[i1] - P[i0], P[i2] - P[i0])
    if np.linalg.norm(nrm) < eps:
        raise ValueError('points are collinear')
    dist = (P - P[i0]) @ nrm
    i3 = int(np.argmax(np.abs(dist)))
    if abs(dist[i3]) < eps * max(1.0, np.linalg.norm(nrm)):
        raise ValueError('points are coplanar')
    a, b, c, e = i0, i1, i2, i3
    if dist[i3] > 0:
        b, c = c, b
    faces = [(a, b, c), (a, c, e), (a, e, b), (b, e, c)]
    # make sure all faces point away from the interior point
    cen = P[[a, b, c, e]].mean(0)
    fixed = []
    for f in faces:
        nn = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
        fixed.append(f if nn @ (P[f[0]] - cen) > 0 else (f[0], f[2], f[1]))
    faces = fixed
    for p in range(n):
        if p in (a, b, c, e):
            continue
        visible = []
        for fi, f in enumerate(faces):
            nn = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
            ln = np.linalg.norm(nn)
            if ln > 0 and nn @ (P[p] - P[f[0]]) / ln > 1e-9:
                visible.append(fi)
        if not visible:
            continue
        vis = set(visible)
        edges = {}
        for fi in visible:
            f = faces[fi]
            for u, w in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                edges[(u, w)] = True
        horizon = [(u, w) for (u, w) in edges if (w, u) not in edges]
        faces = [f for fi, f in enumerate(faces) if fi not in vis]
        for u, w in horizon:
            faces.append((u, w, p))
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    return P[used], np.array([[remap[i] for i in f] for f in faces], dtype=np.int64)


def hull_planes(V, F):
    """Unique outward planes (normal, d) with n.x + d = 0 on the plane."""
    seen = {}
    for f in F:
        nn = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        ln = np.linalg.norm(nn)
        if ln < 1e-12:
            continue
        nn = nn / ln
        d = -float(nn @ V[f[0]])
        key = (tuple(np.round(nn, 4)), round(d, 4))
        seen.setdefault(key, (nn, d))
    return list(seen.values())


def mass_properties(V, F, mass):
    """Solid mass properties of a closed triangle mesh -> (center, 3x3 inertia about the center).

    Falls back to the bounding box when the mesh is open or degenerate."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    bb = V.max(0) - V.min(0)
    box_vol = float(np.prod(np.maximum(bb, 1e-6)))
    vol = 0.0
    com = np.zeros(3)
    C = np.zeros((3, 3))
    if len(F):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vi = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
        vol = float(vi.sum())
        if abs(vol) > 1e-6 * box_vol:
            com = (vi[:, None] * (a + b + c) / 4.0).sum(0) / vol
            s = a + b + c
            C = (vi[:, None, None] / 20.0 * (a[:, :, None] * a[:, None, :] + b[:, :, None] * b[:, None, :] +
                                              c[:, :, None] * c[:, None, :] + s[:, :, None] * s[:, None, :])).sum(0)
            C = C * np.sign(vol)
            vol = abs(vol)
            C = C - vol * np.outer(com, com)
    if abs(vol) <= 1e-6 * box_vol:
        vol = box_vol
        com = (V.max(0) + V.min(0)) / 2
        C = np.diag(bb ** 2 / 12.0) * vol        # second moments of a box about its center
    dens = mass / vol
    I = (np.trace(C) * np.eye(3) - C) * dens
    return com, I


def mopp_matches(props, hv, faces):
    """True when the stored MOPP of an imported packed mesh is still valid for these vertices / triangles."""
    try:
        if 'nif_mopp_data' not in props:
            return False
        v0 = np.frombuffer(base64.b64decode(props['nif_mopp_verts']), dtype='<f4').reshape(-1, 3)
        if v0.shape != hv.shape or not np.allclose(v0, hv, atol=0.02):
            return False
        return hashlib.sha1(np.ascontiguousarray(faces, dtype=np.int32).tobytes()).hexdigest() == props.get('nif_mopp_faces')
    except Exception:                                              # noqa: BLE001
        return False


# --------------------------------------------------------------------- export
class CollisionExporter:
    """Writes Havok collision into an ExportBuilder (``b`` supplies add(), inv, warnings, strip helper)."""

    def __init__(self, b, strip_func):
        self.b = b
        self.strip = strip_func

    # -- small blocks
    @staticmethod
    def filt(layer):
        return {'Layer': int(layer), 'Flags': 0, 'Group': 0}

    @staticmethod
    def mat(m):
        return {'Material': int(m)}

    def export_node(self, node, items, is_skinned_root=False):
        """items: list of (ObjDesc, ExportCollision).  Attaches a bhkCollisionObject to ``node``."""
        b = self.b
        props = items[0][1].props
        layer = int(props.get('nif_hk_layer', 1))
        material = int(props.get('nif_hk_material', 0))

        single = len(items) == 1
        shapes = []
        all_v, all_f = [], []
        body_M = None                                       # rigid-body transform (single item only)
        for o, c in items:
            M = np.asarray(o.matrix, dtype=np.float64).copy()
            M[:3, 3] *= b.inv                               # translation to NIF units
            R = M[:3, :3]
            local = np.asarray(c.vertices, dtype=np.float64) * b.inv
            ident = np.allclose(R, np.eye(3), atol=1e-5) and np.allclose(M[:3, 3], 0, atol=1e-4)
            if single and not ident:
                body_M = M
            else:
                local = local @ R.T + M[:3, 3] if not ident else local
            hv = local / HAVOK_SCALE
            shape = self.make_shape(c, hv, layer, material)
            if shape is None:
                continue
            shapes.append(shape)
            base = sum(len(x) for x in all_v)
            all_v.append(hv)
            if len(c.faces):
                all_f.append(np.asarray(c.faces, dtype=np.int64) + base)
        if not shapes:
            return None
        if len(shapes) == 1:
            top = shapes[0]
        else:
            top = b.add('bhkListShape', **{
                'Num Sub Shapes': len(shapes), 'Sub Shapes': np.array([s.index for s in shapes], dtype=np.int32),
                'Material': self.mat(material), 'Child Shape Property': dict(EMPTY_PROP),
                'Child Filter Property': dict(EMPTY_PROP), 'Num Filters': len(shapes),
                'Filters': [self.filt(layer) for _ in shapes]})
        body = self.make_body(top, props, layer, np.concatenate(all_v), np.concatenate(all_f) if all_f else np.zeros((0, 3), int), body_M)
        flags = int(props.get('nif_hk_col_flags', 41 if layer == 2 else 1))
        col = b.add('bhkCollisionObject', **{'Target': node.index, 'Flags': flags, 'Body': body.index})
        node['Collision Object'] = col.index
        return col

    # -- shapes
    def make_shape(self, c, hv, layer, material):
        b = self.b
        kind = c.kind
        if len(hv) == 0:
            return None
        if kind == 'BOX':
            ctr, R, half = fit_box(hv * HAVOK_SCALE)
            box = b.add('bhkBoxShape', **{'Material': self.mat(material), 'Radius': 0.1, 'Unused 01': BOX_UNUSED,
                                          'Dimensions': _v3(half / HAVOK_SCALE), 'Unused Float': float(half.min() / HAVOK_SCALE)})
            return self.convex_transform(box, R, ctr / HAVOK_SCALE, material)
        if kind == 'SPHERE':
            ctr, r = fit_sphere(hv)
            sph = b.add('bhkSphereShape', **{'Material': self.mat(material), 'Radius': float(r)})
            return self.convex_transform(sph, np.eye(3), ctr, material)
        if kind == 'CAPSULE':
            cap = fit_capsule(hv)
            if cap is None:
                return self.make_shape(type(c)('SPHERE', c.vertices, c.faces, c.props), hv, layer, material)
            p1, p2, r = cap
            return b.add('bhkCapsuleShape', **{'Material': self.mat(material), 'Radius': float(r), 'Unused 01': bytes(8),
                                               'First Point': _v3(p1), 'Radius 1': float(r),
                                               'Second Point': _v3(p2), 'Radius 2': float(r)})
        if kind == 'CONVEX':
            try:
                V, F = convex_hull(hv)
            except ValueError as e:
                b.warnings.append('Convex collision could not be built (%s); using the mesh instead.' % e)
                return self.make_shape(type(c)('MESH', c.vertices, c.faces, c.props), hv, layer, material)
            planes = hull_planes(V, F)
            if len(V) > 65535 or len(planes) > 65535:
                b.warnings.append('Convex collision has too many vertices; simplify it.')
            return b.add('bhkConvexVerticesShape', **{
                'Material': self.mat(material), 'Radius': 0.1,
                'Vertices Property': dict(EMPTY_PROP), 'Normals Property': dict(EMPTY_PROP),
                'Num Vertices': len(V), 'Vertices': np.concatenate([V, np.zeros((len(V), 1))], axis=1).astype(np.float32),
                'Num Normals': len(planes),
                'Normals': np.array([[n[0], n[1], n[2], d] for n, d in planes], dtype=np.float32).reshape(-1, 4)})
        return self.mesh_shape(c, hv, layer, material)

    def convex_transform(self, shape, R, ctr, material):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = ctr
        return self.b.add('bhkConvexTransformShape', **{
            'Shape': shape.index, 'Material': self.mat(material), 'Radius': 0.1, 'Unused 01': CONVEX_TF_UNUSED,
            'Transform': hk_matrix44(M)})

    def mesh_shape(self, c, hv, layer, material):
        b = self.b
        F = np.asarray(c.faces, dtype=np.int64).reshape(-1, 3)
        if len(F) == 0:
            b.warnings.append('A collision mesh has no faces; skipped.')
            return None
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        if not keep.all():
            F = F[keep]
        if len(hv) > 65535:
            raise_ = 'Collision mesh has more than 65535 vertices; split it.'
            b.warnings.append(raise_)
            return None
        if mopp_matches(c.props, hv, F):
            weld = np.frombuffer(base64.b64decode(c.props['nif_mopp_weld']), dtype='<u2') if 'nif_mopp_weld' in c.props else None
            if weld is None or len(weld) != len(F):
                weld = np.zeros(len(F), dtype=np.uint16)
            nrm = np.cross(hv[F[:, 1]] - hv[F[:, 0]], hv[F[:, 2]] - hv[F[:, 0]])
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            ln[ln == 0] = 1
            nrm = nrm / ln
            tris = [{'Triangle': {'v1': int(f[0]), 'v2': int(f[1]), 'v3': int(f[2])}, 'Welding Info': int(w),
                     'Normal': _v3(n)} for f, w, n in zip(F, weld, nrm)]
            data = b.add('hkPackedNiTriStripsData', **{
                'Num Triangles': len(F), 'Triangles': tris, 'Num Vertices': len(hv), 'Vertices': hv.astype(np.float32)})
            packed = b.add('bhkPackedNiTriStripsShape', **{
                'Num Sub Shapes': 1,
                'Sub Shapes': [{'Havok Filter': self.filt(layer), 'Num Vertices': len(hv), 'Material': self.mat(material)}],
                'User Data': 0, 'Unused 01': bytes(4), 'Radius': 0.1, 'Unused 02': bytes(4),
                'Scale': _v4((1, 1, 1)), 'Radius Copy': 0.1, 'Scale Copy': _v4((1, 1, 1)), 'Data': data.index})
            off = [float(x) for x in c.props['nif_mopp_offset'].split()]
            code = base64.b64decode(c.props['nif_mopp_data'])
            return b.add('bhkMoppBvTreeShape', **{
                'Shape': packed.index, 'Unused 01': bytes.fromhex(c.props.get('nif_mopp_unused', MOPP_UNUSED.hex())),
                'Scale': float(c.props.get('nif_mopp_scale', 1.0)),
                'MOPP Code': {'Data Size': len(code), 'Offset': _v4(off[:3], off[3]), 'Data': code}})
        b.warnings.append("Collision mesh has no valid MOPP (new or edited): exported as bhkNiTriStripsShape. "
                          "This works in-game but is slower than a MOPP-based shape.")
        strip = self.strip(F)
        ctr = (hv.min(0) + hv.max(0)) / 2
        rad = float(np.linalg.norm(hv - ctr, axis=1).max())
        sd = b.add('NiTriStripsData', **{
            'Group ID': 0, 'Num Vertices': len(hv), 'Keep Flags': 0, 'Compress Flags': 0, 'Has Vertices': True,
            'Vertices': hv.astype(np.float32), 'Data Flags': 0, 'Has Normals': False,
            'Bounding Sphere': {'Center': _v3(ctr), 'Radius': rad}, 'Has Vertex Colors': False,
            'Consistency Flags': 0x4000, 'Additional Data': -1,
            'Num Triangles': len(F), 'Num Strips': 1, 'Strip Lengths': np.array([len(strip)], dtype=np.uint16),
            'Has Points': True, 'Points': [strip.astype(np.uint16)]})
        return b.add('bhkNiTriStripsShape', **{
            'Material': self.mat(material), 'Radius': 0.1, 'Unused 01': bytes(20), 'Grow By': 1,
            'Scale': _v4((1, 1, 1)), 'Num Strips Data': 1, 'Strips Data': np.array([sd.index], dtype=np.int32),
            'Num Filters': 1, 'Filters': [self.filt(layer)]})

    # -- rigid body
    def make_body(self, shape, props, layer, hv, faces, body_M):
        b = self.b
        mass = float(props.get('nif_hk_mass', 10.0))
        if mass < 0.0001:
            mass = 0.05 if int(props.get('nif_hk_motion', 7)) != 7 else mass
        center, inertia = mass_properties(hv, faces, max(mass, 0.0001))
        col_filter = self.filt(layer)
        info = {
            'Unused 01': RB_UNUSED01, 'Havok Filter': col_filter, 'Unused 02': RB_UNUSED02,
            'Collision Response': 1, 'Unused 03': 246, 'Process Contact Callback Delay': 65535, 'Unused 04': RB_UNUSED04,
            'Translation': _v4((0, 0, 0)), 'Rotation': _v4((0, 0, 0), 1.0),
            'Linear Velocity': _v4((0, 0, 0)), 'Angular Velocity': _v4((0, 0, 0)),
            'Inertia Tensor': {'m%d%d' % (i + 1, j + 1): (float(inertia[i][j]) if i < 3 and j < 3 else 0.0)
                               for i in range(3) for j in range(4)},
            'Center': _v4(center), 'Mass': mass,
            'Linear Damping': float(props.get('nif_hk_linear_damping', 0.1)),
            'Angular Damping': float(props.get('nif_hk_angular_damping', 0.05)),
            'Friction': float(props.get('nif_hk_friction', 0.3)),
            'Restitution': float(props.get('nif_hk_restitution', 0.3)),
            'Max Linear Velocity': 250.0, 'Max Angular Velocity': 31.4159, 'Penetration Depth': 0.15,
            'Motion System': int(props.get('nif_hk_motion', 7)),
            'Deactivator Type': int(props.get('nif_hk_deactivator', 1)),
            'Solver Deactivation': int(props.get('nif_hk_solver_deactivation', 1)),
            'Quality Type': int(props.get('nif_hk_quality', 1)), 'Unused 05': RB_UNUSED05}
        btype = 'bhkRigidBody'
        if body_M is not None:
            btype = 'bhkRigidBodyT'
            q = matrix_to_quat(body_M[:3, :3] / (np.cbrt(abs(np.linalg.det(body_M[:3, :3]))) or 1.0))
            info['Translation'] = _v4(body_M[:3, 3] / HAVOK_SCALE)
            info['Rotation'] = {'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2]), 'w': float(q[3])}
        return b.add(btype, **{
            'Shape': shape.index, 'Havok Filter': col_filter,
            'World Object Info': {'Unused 01': bytes(4), 'Broad Phase Type': 242, 'Unused 02': RB_WORLD_UNUSED2,
                                  'Property': dict(EMPTY_PROP)},
            'Entity Info': {'Collision Response': 1, 'Unused 01': 0, 'Process Contact Callback Delay': 65535},
            'Rigid Body Info': info, 'Num Constraints': 0, 'Constraints': np.zeros(0, dtype=np.int32), 'Body Flags': 0})
