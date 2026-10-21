# SPDX-License-Identifier: GPL-3.0-or-later
"""
Builds Oblivion NIF blocks from :class:`nif_scene.ObjDesc` objects (the same
description the importer produces, plus ``ObjDesc.xmesh`` for geometry read from
Blender).  No ``bpy`` here, so it is unit-testable.

Output NIF: version 20.0.0.5, user version 11, BS version 11.
"""
import os
from dataclasses import dataclass

import numpy as np

from .nif_reader import Block
from .nif_scene import MaterialDesc, ExportMesh, bone_axis_matrix
from .nif_collision import CollisionExporter
from .nif_writer import new_block, nif_bytes

MAX_BONES_PER_PARTITION = 18
MAX_WEIGHTS_PER_VERTEX = 4


class NifExportError(Exception):
    pass


@dataclass
class ExportSettings:
    scale: float = 0.1                # NIF -> Blender scale used on import (positions are divided by this)
    tangents: str = 'AUTO'            # 'AUTO'/'ALWAYS' (whenever UVs+normals exist) | 'NEVER'
    author: str = 'Blender Oblivion NIF exporter'
    collision: bool = True            # export Havok collision helpers
    prn: str = 'NONE'                 # weapon/armour attach point: NONE BACK SIDE QUIVER SHIELD HELM RING


# ------------------------------------------------------------------- helpers
def vec(v):
    return {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2])}


def rot_dict(A):
    """3x3 column-vector rotation -> NIF Matrix33 fields (rows = file chunks)."""
    return {'m11': float(A[0][0]), 'm21': float(A[0][1]), 'm31': float(A[0][2]),
            'm12': float(A[1][0]), 'm22': float(A[1][1]), 'm32': float(A[1][2]),
            'm13': float(A[2][0]), 'm23': float(A[2][1]), 'm33': float(A[2][2])}


def split_matrix(M, inv_scale):
    """4x4 (Blender units) -> (translation, rotation3x3, uniform scale) in NIF units."""
    A = np.asarray(M, dtype=np.float64)[:3, :3]
    sc = float(np.cbrt(abs(np.linalg.det(A)))) or 1.0
    return np.asarray(M, dtype=np.float64)[:3, 3] * inv_scale, A / sc, sc


def to_nif_matrix(M, inv_scale):
    t, R, sc = split_matrix(M, inv_scale)
    out = np.eye(4)
    out[:3, :3] = R * sc
    out[:3, 3] = t
    return out


def transform_dict(M4):
    A = M4[:3, :3]
    sc = float(np.cbrt(abs(np.linalg.det(A)))) or 1.0
    return {'Rotation': rot_dict(A / sc), 'Translation': vec(M4[:3, 3]), 'Scale': sc}


def nif_texture_path(p):
    p = p.replace('/', '\\')
    low = p.lower()
    i = low.rfind('textures\\')
    if i >= 0:
        return p[i:]
    return 'textures\\' + os.path.basename(p)


def _canon(t):
    a, b, c = t
    return (a, b, c) if a <= b and a <= c else ((b, c, a) if b <= c else (c, a, b))


def greedy_strips(tris):
    """Split a triangle list into triangle strips (winding preserved).  Returns a list of index lists."""
    tris = [tuple(int(x) for x in t) for t in tris]
    n = len(tris)
    edge = {}
    for i, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            edge.setdefault((min(u, v), max(u, v)), []).append(i)
    used = [False] * n

    def free_neighbours(i):
        a, b, c = tris[i]
        return sum(1 for u, v in ((a, b), (b, c), (c, a))
                   for j in edge[(min(u, v), max(u, v))] if j != i and not used[j])

    def extend(strip, taken):
        """Greedily extend ``strip`` (list of verts); ``taken`` collects triangle ids."""
        while True:
            u, v = strip[-2], strip[-1]
            m = len(strip) - 2                       # index of the triangle the next vertex would create
            best = None
            for j in edge.get((min(u, v), max(u, v)), ()):
                if used[j] or j in taken:
                    continue
                t = tris[j]
                for w in t:
                    if w == u or w == v:
                        continue
                    want = (u, v, w) if m % 2 == 0 else (u, w, v)
                    if _canon(want) == _canon(t):
                        best = (j, w)
                        break
                if best:
                    break
            if best is None:
                return
            taken.append(best[0])
            strip.append(best[1])

    order = sorted(range(n), key=lambda i: free_neighbours(i))
    strips = []
    for i in order:
        if used[i]:
            continue
        a, b, c = tris[i]
        cand = None
        for start in ((a, b, c), (b, c, a), (c, a, b)):
            st, tk = list(start), [i]
            extend(st, tk)
            if cand is None or len(st) > len(cand[0]):
                cand = (st, tk)
        st, tk = cand
        for j in tk:
            used[j] = True
        strips.append(st)
    return strips


def strip_from_triangles(tris):
    """Triangle list -> ONE stitched strip (degenerate joins keep the winding)."""
    strip = []
    for s in greedy_strips(tris):
        if not strip:
            strip = list(s)
            continue
        a = s[0]
        strip += [strip[-1], a, a]
        if (len(strip) - 1) % 2 == 1:         # the real triangle must start on an even index (winding)
            strip.append(a)
        strip += s[1:3]
        strip += s[3:]
    return np.array(strip, dtype=np.uint16)


def compute_tangents(pos, nrm, uv, tris):
    """Per-vertex tangent / bitangent from UV0 (NIF uv orientation)."""
    n = len(pos)
    tan = np.zeros((n, 3))
    bit = np.zeros((n, 3))
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    u0, u1, u2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = u1 - u0, u2 - u0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det[np.abs(det) < 1e-12] = 1e-12
    r = 1.0 / det
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    for k in range(3):
        np.add.at(tan, tris[:, k], t)
        np.add.at(bit, tris[:, k], b)
    tan = tan - nrm * np.sum(nrm * tan, axis=1, keepdims=True)
    ln = np.linalg.norm(tan, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    tan /= ln
    sign = np.where(np.sum(np.cross(nrm, tan) * bit, axis=1) < 0, -1.0, 1.0)[:, None]
    bit2 = np.cross(nrm, tan) * sign
    return tan.astype(np.float32), bit2.astype(np.float32)


# ------------------------------------------------------------------ builder
class ExportBuilder:
    def __init__(self, objs, settings: ExportSettings):
        self.objs = objs
        self.colls = []
        self.has_collision = False
        self.node_of = {}
        self.s = settings
        self.inv = 1.0 / settings.scale
        self.blocks = []
        self.warnings = []
        self.children = {}
        self._mat_blocks = {}
        self.bone_blocks = {}      # (armature id, bone name) -> NiNode block
        self.bone_rest = {}        # (armature id, bone name) -> 4x4 NIF rest matrix in skeleton space
        self.arm_block = {}        # id(armature obj) -> NiNode block

    def add(self, btype, **fields):
        b = new_block(btype, **fields)
        b.index = len(self.blocks)
        self.blocks.append(b)
        return b

    # ---- common AV object fields ------------------------------------------
    def av_fields(self, name, M4, extras=(), props=(), flags=14):
        return dict(**{'Name': name,
                       'Num Extra Data List': len(extras),
                       'Extra Data List': np.array([e.index for e in extras], dtype=np.int32),
                       'Controller': -1, 'Flags': flags,
                       'Translation': vec(M4[:3, 3]),
                       'Rotation': rot_dict(M4[:3, :3] / (np.cbrt(abs(np.linalg.det(M4[:3, :3]))) or 1.0)),
                       'Scale': (lambda k: 1.0 if abs(k - 1.0) < 1e-5 else k)(float(np.cbrt(abs(np.linalg.det(M4[:3, :3]))) or 1.0)),
                       'Num Properties': len(props),
                       'Properties': np.array([p.index for p in props], dtype=np.int32),
                       'Collision Object': -1})

    def node_fields(self, name, M4, extras=(), flags=14):
        d = self.av_fields(name, M4, extras, (), flags)
        d.update({'Num Children': 0, 'Children': np.zeros(0, np.int32), 'Num Effects': 0,
                  'Effects': np.zeros(0, np.int32)})
        return d

    @staticmethod
    def set_children(node, children):
        node['Num Children'] = len(children)
        node['Children'] = np.array([c.index for c in children], dtype=np.int32)

    PRN = {'BACK': 'BackWeapon', 'SIDE': 'SideWeapon', 'QUIVER': 'Quiver', 'SHIELD': 'Bip01 L ForearmTwist',
           'HELM': 'Bip01 Head', 'RING': 'Bip01 R Finger1'}

    def extra_blocks(self, o, is_root=False):
        """Extra-data blocks restored from the object's nif_* custom properties (+ BSX/BSBound/Prn on the root)."""
        out = []
        if is_root:
            bsx_value = 0
            for q in self.objs:
                if 'nif_bsx_flags' in q.props:
                    bsx_value = int(q.props['nif_bsx_flags'])
                    break
            if self.has_collision:
                bsx_value |= 2                       # Havok flag (the 2.49 exporter's default for collision files)
            out.append(self.add('BSXFlags', **{'Name': 'BSX', 'Integer Data': bsx_value}))
            bound = next((q.props['nif_bsbound'] for q in self.objs if 'nif_bsbound' in q.props), None)
            if bound:
                try:
                    v = [float(x) for x in str(bound).split()]
                    out.append(self.add('BSBound', **{'Name': 'BBX', 'Center': tuple(v[:3]), 'Dimensions': tuple(v[3:6])}))
                except (ValueError, TypeError):
                    self.warnings.append('Ignoring malformed nif_bsbound property: %r' % (bound,))
        have_prn = False
        for k, v in sorted(o.props.items()):
            if not k.startswith('nif_extra_'):
                continue
            name = k[len('nif_extra_'):]
            have_prn |= name.lower() == 'prn'
            if isinstance(v, bool) or isinstance(v, int):
                out.append(self.add('NiIntegerExtraData', **{'Name': name, 'Integer Data': int(v) & 0xFFFFFFFF}))
            elif isinstance(v, float):
                out.append(self.add('NiFloatExtraData', **{'Name': name, 'Float Data': float(v)}))
            else:
                out.append(self.add('NiStringExtraData', **{'Name': name, 'String Data': str(v)}))
        if is_root and self.s.prn in self.PRN and not have_prn:
            out.append(self.add('NiStringExtraData', **{'Name': 'Prn', 'String Data': self.PRN[self.s.prn]}))
        return out

    # ---- main --------------------------------------------------------------
    def build(self):
        objs = [o for o in self.objs if o.collection == 'main']
        self.colls = [o for o in self.objs if o.collection == 'collision' and o.xcoll is not None] if self.s.collision else []
        self.has_collision = bool(self.colls)
        self.node_of = {}
        for o in objs:
            self.children.setdefault(id(o.parent) if o.parent is not None else None, []).append(o)
        tops = self.children.get(None, [])
        if not tops:
            raise NifExportError('Nothing to export.')
        # root node: reuse a lone top-level node/armature, otherwise synthesise one
        if len(tops) == 1 and tops[0].kind in ('EMPTY', 'ARMATURE'):
            root = self.add('NiNode')
            self.fill_container(root, tops[0], is_root=True)
        else:
            root = self.add('NiNode')
            root.update(self.node_fields('Scene Root', np.eye(4), self.extra_blocks(tops[0], True), 2))
            kids = [k for k in (self.emit_obj(o) for o in tops) if k is not None]
            self.set_children(root, kids)
        self.export_collision(root)
        return self.blocks, [root]

    def export_collision(self, root):
        if not self.colls:
            return
        groups = {}
        for o in self.colls:
            p = o.parent
            while p is not None and id(p) not in self.node_of:
                p = p.parent
            node = self.node_of.get(id(p), root)
            groups.setdefault(node.index, (node, []))[1].append((o, o.xcoll))
        ce = CollisionExporter(self, strip_from_triangles)
        for node, items in groups.values():
            ce.export_node(node, items)

    def fill_container(self, node, obj, is_root=False):
        M = to_nif_matrix(obj.matrix, self.inv)
        node.update(self.node_fields(obj.name or 'Scene Root', M, self.extra_blocks(obj, is_root), 2 if is_root else 14))
        self.node_of[id(obj)] = node
        kids = self.emit_children(obj, node)
        self.set_children(node, kids)

    def emit_children(self, obj, node):
        kids = []
        bones = []
        if obj.kind == 'ARMATURE':
            self.arm_block[id(obj)] = node
            bones = self.make_bones(obj)
        # (bones are emitted first so skin blocks can reference them, but listed after the geometry like vanilla)
        for c in self.children.get(id(obj), []):
            k = self.emit_obj(c)
            if k is not None:
                kids.append(k)
        return kids + bones

    def emit_obj(self, o):
        if o.kind == 'EMPTY':
            node = self.add('NiNode')
            node.update(self.node_fields(o.name, to_nif_matrix(o.matrix, self.inv), self.extra_blocks(o)))
            self.node_of[id(o)] = node
            self.set_children(node, self.emit_children(o, node))
            return node
        if o.kind == 'ARMATURE':
            node = self.add('NiNode')
            node.update(self.node_fields(o.name, to_nif_matrix(o.matrix, self.inv), self.extra_blocks(o), 2))
            self.node_of[id(o)] = node
            self.set_children(node, self.emit_children(o, node))
            return node
        if o.kind == 'MESH' and o.xmesh is not None:
            return self.make_shape(o)
        return None

    # ---- bones ----------------------------------------------------------------
    def make_bones(self, arm):
        fwd = int(arm.props.get('nif_bone_forward', 0))
        C = bone_axis_matrix(fwd)
        rest = {}
        for b in arm.bones:
            M = np.eye(4)
            M[:3, :3] = np.asarray(b.matrix)[:3, :3] @ C.T
            M[:3, 3] = np.asarray(b.matrix)[:3, 3] * self.inv
            rest[b.name] = M
            self.bone_rest[(id(arm), b.name)] = M
        blocks = {}
        for b in arm.bones:
            n = self.add('NiNode')
            blocks[b.name] = n
            self.bone_blocks[(id(arm), b.name)] = n
        tops = []
        kids = {}
        for b in arm.bones:
            if b.parent and b.parent in blocks and b.parent != b.name:
                local = np.linalg.inv(rest[b.parent]) @ rest[b.name]
                kids.setdefault(b.parent, []).append(blocks[b.name])
            else:
                local = rest[b.name]
                tops.append(blocks[b.name])
            blocks[b.name].update(self.node_fields(b.name, local, [], 16))
        for pname, lst in kids.items():
            self.set_children(blocks[pname], lst)
        return tops

    # ---- materials ---------------------------------------------------------------
    def material_blocks(self, md: MaterialDesc):
        key = id(md)
        if key in self._mat_blocks:
            return self._mat_blocks[key]
        props = []
        if md.specular_enabled:
            props.append(self.add('NiSpecularProperty', **self.prop_base('', Flags=1)))
        props.append(self.add('NiMaterialProperty', **self.prop_base(
            md.name, **{'Ambient Color': dict(r=md.ambient[0], g=md.ambient[1], b=md.ambient[2]),
                        'Diffuse Color': dict(r=md.diffuse[0], g=md.diffuse[1], b=md.diffuse[2]),
                        'Specular Color': dict(r=md.specular[0], g=md.specular[1], b=md.specular[2]),
                        'Emissive Color': dict(r=md.emissive[0], g=md.emissive[1], b=md.emissive[2]),
                        'Glossiness': float(md.glossiness), 'Alpha': float(md.alpha)})))
        slots = {'base': 'Base Texture', 'dark': 'Dark Texture', 'detail': 'Detail Texture',
                 'gloss': 'Gloss Texture', 'glow': 'Glow Texture', 'decal0': 'Decal 0 Texture'}
        tex_fields = {}
        for slot, field in slots.items():
            ts = md.textures.get(slot)
            if ts is None:
                continue
            src = self.add('NiSourceTexture', **self.prop_base(
                '', **{'Use External': 1, 'File Name': nif_texture_path(ts.path), 'Pixel Data': -1,
                       'Format Prefs': {'Pixel Layout': 6, 'Use Mipmaps': 1, 'Alpha Format': 3},
                       'Is Static': 1, 'Direct Render': True}))
            tex_fields['Has ' + field] = True
            tex_fields[field] = {'Source': src.index, 'Clamp Mode': ts.clamp_mode, 'Filter Mode': ts.filter_mode,
                                 'UV Set': ts.uv_set, 'Has Texture Transform': False}
        if tex_fields:
            tex_fields.update({'Apply Mode': 2, 'Texture Count': 7, 'Num Shader Textures': 0, 'Shader Textures': []})
            props.append(self.add('NiTexturingProperty', **self.prop_base('', **tex_fields)))
        if md.alpha_blend or md.alpha_test:
            fl = (1 if md.alpha_blend else 0) | (md.alpha_src << 1) | (md.alpha_dst << 5)
            if md.alpha_test:
                fl |= 0x200 | (4 << 10)
            props.append(self.add('NiAlphaProperty', **self.prop_base(
                '', Flags=fl, Threshold=int(round(md.alpha_threshold * 255)))))
        if md.two_sided:
            props.append(self.add('NiStencilProperty', **self.prop_base(
                '', **{'Stencil Enabled': 0, 'Stencil Function': 0, 'Stencil Ref': 0, 'Stencil Mask': 0xFFFFFFFF,
                       'Fail Action': 0, 'Z Fail Action': 0, 'Pass Action': 0, 'Draw Mode': 3})))
        self._mat_blocks[key] = props
        return props

    @staticmethod
    def prop_base(name, **extra):
        d = {'Name': name, 'Num Extra Data List': 0, 'Extra Data List': np.zeros(0, np.int32), 'Controller': -1}
        d.update(extra)
        return d

    # ---- geometry -----------------------------------------------------------------
    def make_shape(self, o):
        xm: ExportMesh = o.xmesh
        md = xm.material or MaterialDesc(key=('default',), name='Default')
        nc = len(xm.corner_vertex)
        if nc == 0 or nc % 3:
            self.warnings.append("Mesh '%s' has no triangles; skipped." % o.name)
            return None
        cv = np.asarray(xm.corner_vertex, dtype=np.int64)
        pos = np.asarray(xm.positions, dtype=np.float64) * self.inv
        # ---- vertex splitting
        cols = [cv[:, None].astype(np.float64)]
        nrm = None
        if xm.corner_normals is not None:
            nrm = np.asarray(xm.corner_normals, dtype=np.float64)
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            ln[ln == 0] = 1.0
            nrm = nrm / ln
            cols.append(np.round(nrm, 4))
        uvs = []
        for uv in xm.corner_uvs:
            u = np.asarray(uv, dtype=np.float64).copy()
            u[:, 1] = 1.0 - u[:, 1]
            uvs.append(u)
            cols.append(np.round(u, 5))
        col = None
        if xm.corner_colors is not None:
            col = np.asarray(xm.corner_colors, dtype=np.float64)
            cols.append(np.round(col, 4))
        key = np.concatenate(cols, axis=1)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(first)
        remap = np.empty(len(order), dtype=np.int64)
        remap[order] = np.arange(len(order))
        corner_to_nif = remap[inv]
        src_corner = first[order]
        nv = len(order)
        if nv > 65535:
            raise NifExportError("Mesh '%s' needs %d vertices after splitting; Oblivion's limit is 65535. "
                                 "Split the mesh." % (o.name, nv))
        tris = corner_to_nif.reshape(-1, 3)
        keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
        tris = tris[keep]
        verts = pos[cv[src_corner]]
        normals = nrm[src_corner] if nrm is not None else None
        if normals is None and len(tris):                    # no normals supplied: smooth area-weighted
            fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
            normals = np.zeros_like(verts)
            for k in range(3):
                np.add.at(normals, tris[:, k], fn)
            ln = np.linalg.norm(normals, axis=1, keepdims=True)
            ln[ln == 0] = 1.0
            normals = normals / ln
        uv_arrays = [u[src_corner] for u in uvs]
        colors = col[src_corner] if col is not None else None

        center = (verts.min(0) + verts.max(0)) / 2
        radius = float(np.linalg.norm(verts - center, axis=1).max())
        data = self.add('NiTriShapeData')
        data.update({
            'Group ID': 0, 'Num Vertices': nv, 'Keep Flags': 0, 'Compress Flags': 0, 'Has Vertices': True,
            'Vertices': verts.astype(np.float32), 'Data Flags': len(uv_arrays),
            'Has Normals': normals is not None,
            'Bounding Sphere': {'Center': vec(center), 'Radius': radius},
            'Has Vertex Colors': colors is not None,
            'Consistency Flags': 0x4000, 'Additional Data': -1,
            'Num Triangles': len(tris), 'Num Triangle Points': len(tris) * 3, 'Has Triangles': True,
            'Triangles': tris.astype(np.uint16), 'Num Match Groups': 0, 'Match Groups': []})
        if normals is not None:
            data['Normals'] = normals.astype(np.float32)
        if colors is not None:
            data['Vertex Colors'] = colors.astype(np.float32)
        if uv_arrays:
            data['UV Sets'] = np.stack(uv_arrays).astype(np.float32)

        # ---- tangent space extra data
        extras = []
        want = self.s.tangents != 'NEVER'  # 'AUTO' = always when UVs + normals exist (as the 2.49 exporter did)
        if want and normals is not None and uv_arrays and len(tris):
            t, b = compute_tangents(verts, normals, uv_arrays[0], tris)
            # bitangents first, then tangents (verified on vanilla meshes)
            blob = np.concatenate([b, t], axis=0).astype('<f4').tobytes()
            extras.append(self.add('NiBinaryExtraData', **{
                'Name': 'Tangent space (binormal & tangent vectors)',
                'Binary Data': {'Data Size': len(blob), 'Data': blob}}))

        extras += self.extra_blocks(o)
        props = self.material_blocks(md)
        shape = self.add('NiTriShape')
        M = to_nif_matrix(o.matrix, self.inv)
        shape.update(self.av_fields(o.name, M, extras, props, 0 if (xm.weights and o.armature is not None) else 14))
        shape.update({'Data': data.index, 'Skin Instance': -1, 'Material Data': {'Has Shader': False}})

        if xm.weights and o.armature is not None:
            self.make_skin(shape, o, xm, M, verts, cv[src_corner], tris)
        return shape

    # ---- skinning -------------------------------------------------------------------
    def make_skin(self, shape, o, xm, G, verts, src_vertex, tris):
        arm = o.armature
        if id(arm) not in self.arm_block:
            self.warnings.append("Mesh '%s': armature was not exported; skin skipped." % o.name)
            return
        # dense per-Blender-vertex weight table, keep the strongest 4
        names = [n for n in xm.weights if (id(arm), n) in self.bone_blocks]
        if not names:
            self.warnings.append("Mesh '%s': no vertex group matches a bone; skin skipped." % o.name)
            return
        nbv = len(xm.positions)
        W = np.zeros((nbv, len(names)))
        for k, nme in enumerate(names):
            idx, w = xm.weights[nme]
            np.add.at(W[:, k], np.asarray(idx, dtype=np.int64), np.asarray(w, dtype=np.float64))
        top = np.argsort(-W, axis=1)[:, :MAX_WEIGHTS_PER_VERTEX]
        tw = np.take_along_axis(W, top, axis=1)
        if top.shape[1] < MAX_WEIGHTS_PER_VERTEX:           # fewer bones than slots: pad with empty slots
            pad = MAX_WEIGHTS_PER_VERTEX - top.shape[1]
            top = np.pad(top, ((0, 0), (0, pad)))
            tw = np.pad(tw, ((0, 0), (0, pad)))
        tw[tw < 1e-4] = 0
        sums = tw.sum(axis=1, keepdims=True)
        unweighted = (sums[:, 0] == 0)
        if unweighted[src_vertex].any():
            self.warnings.append("Mesh '%s': %d vertices have no bone weights." % (
                o.name, int(unweighted[src_vertex].sum())))
        sums[sums == 0] = 1.0
        tw = tw / sums
        # per NIF vertex
        nif_top = top[src_vertex]
        nif_w = tw[src_vertex].astype(np.float32)
        used_cols = sorted(set(nif_top[nif_w > 0].tolist()))
        if not used_cols:
            return
        bone_names = [names[c] for c in used_cols]
        col_to_local = {c: i for i, c in enumerate(used_cols)}
        local_idx = np.vectorize(lambda c: col_to_local.get(c, 0))(nif_top).astype(np.int64)

        skel_root = self.arm_block[id(arm)]
        bone_list = []
        for li, c in enumerate(used_cols):
            sel = (nif_top == c) & (nif_w > 0)
            vi, slot = np.nonzero(sel)
            wts = nif_w[vi, slot]
            bind = self.bone_rest[(id(arm), names[c])]
            off = np.linalg.inv(bind) @ G
            pts = verts[vi] if len(vi) else np.zeros((1, 3))
            cen = (pts.min(0) + pts.max(0)) / 2
            rad = float(np.linalg.norm(pts - cen, axis=1).max())
            # the per-bone bounding sphere lives in BONE space (like vanilla), not mesh space
            cen = off[:3, :3] @ cen + off[:3, 3]
            vw = np.zeros(len(vi), dtype=[('Index', '<u2'), ('Weight', '<f4')])
            vw['Index'] = vi
            vw['Weight'] = wts
            bone_list.append({'Skin Transform': transform_dict(off),
                              'Bounding Sphere': {'Center': vec(cen), 'Radius': rad},
                              'Num Vertices': len(vi), 'Vertex Weights': vw})
        sd = self.add('NiSkinData')
        sd.update({'Skin Transform': transform_dict(G), 'Num Bones': len(bone_list),
                   'Has Vertex Weights': True, 'Bone List': bone_list})
        sp = self.add('NiSkinPartition')
        sp.update(self.make_partitions(tris, nif_top, nif_w, local_idx, len(verts)))
        si = self.add('NiSkinInstance')
        si.update({'Data': sd.index, 'Skin Partition': sp.index, 'Skeleton Root': skel_root.index,
                   'Num Bones': len(bone_names),
                   'Bones': np.array([self.bone_blocks[(id(arm), n)].index for n in bone_names], dtype=np.int32)})
        shape['Skin Instance'] = si.index

    def make_partitions(self, tris, nif_top, nif_w, local_idx, nverts):
        # per-vertex local bone sets
        vbones = [set(local_idx[v][nif_w[v] > 0].tolist()) for v in range(nverts)]
        parts = []                         # list of (bone set, [triangle rows])
        cur_b, cur_t = set(), []
        for ti, tri in enumerate(tris):
            need = vbones[tri[0]] | vbones[tri[1]] | vbones[tri[2]]
            if len(cur_b | need) > MAX_BONES_PER_PARTITION and cur_t:
                parts.append((cur_b, cur_t))
                cur_b, cur_t = set(), []
            cur_b |= need
            cur_t.append(ti)
        if cur_t:
            parts.append((cur_b, cur_t))
        out = []
        for bset, tl in parts:
            bones = sorted(bset) or [0]
            b2p = {b: i for i, b in enumerate(bones)}
            ptris = tris[tl]
            vmap = np.unique(ptris)
            v2p = {int(v): i for i, v in enumerate(vmap)}
            lt = np.vectorize(lambda v: v2p[int(v)])(ptris).reshape(-1, 3)
            weights = np.zeros((len(vmap), MAX_WEIGHTS_PER_VERTEX), dtype=np.float32)
            bidx = np.zeros((len(vmap), MAX_WEIGHTS_PER_VERTEX), dtype=np.uint8)
            for i, v in enumerate(vmap):
                for k in range(MAX_WEIGHTS_PER_VERTEX):
                    if nif_w[v, k] > 0:
                        weights[i, k] = nif_w[v, k]
                        bidx[i, k] = b2p[int(local_idx[v, k])]
            strip = strip_from_triangles(lt)
            out.append({'Num Vertices': len(vmap), 'Num Triangles': len(strip) - 2, 'Num Bones': len(bones),
                        'Num Strips': 1, 'Num Weights Per Vertex': MAX_WEIGHTS_PER_VERTEX,
                        'Bones': np.array(bones, dtype=np.uint16), 'Has Vertex Map': True,
                        'Vertex Map': vmap.astype(np.uint16), 'Has Vertex Weights': True,
                        'Vertex Weights': weights, 'Strip Lengths': np.array([len(strip)], dtype=np.uint16),
                        'Has Faces': True, 'Strips': [strip], 'Has Bone Indices': True, 'Bone Indices': bidx})
        return {'Num Partitions': len(out), 'Partitions': out}


def build_export(objs, settings=None):
    b = ExportBuilder(objs, settings or ExportSettings())
    blocks, roots = b.build()
    return blocks, roots, b.warnings


def export_bytes(objs, settings=None):
    settings = settings or ExportSettings()
    blocks, roots, warnings = build_export(objs, settings)
    data, _ = nif_bytes(blocks, roots, author=settings.author)
    return data, warnings


def xmesh_from_meshdesc(md):
    """Importer MeshDesc (per-vertex) -> ExportMesh (per-corner), used for round-trip tests."""
    cv = md.faces.reshape(-1).astype(np.int64)
    uvs = [uv[cv] for uv in md.uv_sets]
    return ExportMesh(md.name, md.vertices, cv,
                      md.normals[cv] if md.normals is not None else None, uvs,
                      md.colors[cv] if md.colors is not None else None, md.material,
                      md.skin.weights if md.skin is not None else None)
