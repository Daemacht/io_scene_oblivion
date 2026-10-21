# SPDX-License-Identifier: GPL-3.0-or-later
"""
Converts a parsed :class:`nif_reader.NifFile` into plain Python / numpy
descriptions of what should exist in Blender (objects, meshes, materials,
armatures).  This module does *not* import ``bpy`` so it can be unit tested
outside of Blender; ``blender_build.py`` turns the result into real data blocks.

Conventions
-----------
* NIF is right-handed Z-up like Blender, so no axis conversion is needed.
* NIF rotation matrices are stored as three consecutive rows of a
  column-vector rotation matrix (verified against skin bind poses).
* ``settings.scale`` converts NIF units to Blender units.  It is applied to
  positions here, never to rotations.
* UVs are flipped (v -> 1 - v).
* Skinned bind pose: bone_rest = geometry_world * inverse(bone_skin_transform).
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

HAVOK_SCALE = 7.0   # 1 Havok unit == 7 NIF units in Oblivion


# ------------------------------------------------------------------ settings
@dataclass
class ImportSettings:
    scale: float = 0.1
    texture_dir: str = ''              # game "Data" folder (or any folder containing "textures")
    import_collision: bool = False
    import_skeleton: bool = True       # build Armature for skinned meshes / skeleton files
    import_normals: bool = True
    import_vertex_colors: bool = True
    import_hidden: bool = True
    find_normal_maps: bool = True      # look for <name>_n.dds next to the diffuse
    import_empties: bool = True        # keep NiNode hierarchy as empties


# --------------------------------------------------------------- description
@dataclass
class TextureSlot:
    slot: str                          # base, dark, detail, gloss, glow, bump, normal, decal0..
    path: str                          # path as stored in the NIF
    resolved: Optional[str] = None     # absolute path on disk if found
    uv_set: int = 0
    clamp_mode: int = 3
    filter_mode: int = 2
    transform: Optional[dict] = None


@dataclass
class MaterialDesc:
    key: tuple
    name: str
    diffuse: tuple = (1.0, 1.0, 1.0)
    ambient: tuple = (1.0, 1.0, 1.0)
    specular: tuple = (0.0, 0.0, 0.0)
    emissive: tuple = (0.0, 0.0, 0.0)
    glossiness: float = 10.0
    alpha: float = 1.0
    specular_enabled: bool = False
    textures: Dict[str, TextureSlot] = field(default_factory=dict)
    alpha_blend: bool = False
    alpha_test: bool = False
    alpha_threshold: float = 0.5
    alpha_src: int = 6
    alpha_dst: int = 7
    two_sided: bool = False
    props: dict = field(default_factory=dict)


@dataclass
class SkinDesc:
    # bone name -> (vertex indices uint array, weights float array)
    weights: Dict[str, tuple] = field(default_factory=dict)


@dataclass
class MeshDesc:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uv_sets: List[np.ndarray] = field(default_factory=list)
    colors: Optional[np.ndarray] = None
    material: Optional[MaterialDesc] = None
    skin: Optional[SkinDesc] = None
    convex_hull: bool = False          # build hull from vertices (Havok convex shapes)


@dataclass
class ExportMesh:
    """Mesh as read from Blender: per-triangle-corner data (3 corners per triangle)."""
    name: str
    positions: np.ndarray                      # (nv,3) Blender units, object-local
    corner_vertex: np.ndarray                  # (3*nt,) vertex index per corner
    corner_normals: Optional[np.ndarray] = None
    corner_uvs: List[np.ndarray] = field(default_factory=list)   # Blender convention (v up)
    corner_colors: Optional[np.ndarray] = None
    material: Optional['MaterialDesc'] = None
    weights: Optional[Dict[str, tuple]] = None   # bone name -> (vertex idx, weights)


@dataclass
class BoneDesc:
    name: str
    parent: Optional[str]
    matrix: np.ndarray                 # 4x4 edit-bone matrix, armature space, Blender units
    length: float
    nif_matrix: Optional[np.ndarray] = None


@dataclass
class ExportCollision:
    """Collision geometry as read from Blender (export side)."""
    kind: str                          # MESH | BOX | SPHERE | CAPSULE | CONVEX
    vertices: np.ndarray               # (n,3) Blender units, object-local, object scale applied
    faces: np.ndarray                  # (m,3) int
    props: dict = field(default_factory=dict)


COLLISION_KINDS = {'bhkBoxShape': 'BOX', 'bhkSphereShape': 'SPHERE', 'bhkMultiSphereShape': 'SPHERE',
                   'bhkCapsuleShape': 'CAPSULE', 'bhkConvexVerticesShape': 'CONVEX',
                   'BOX': 'BOX', 'SPHERE': 'SPHERE', 'CAPSULE': 'CAPSULE', 'CONVEX': 'CONVEX', 'MESH': 'MESH'}


def collision_from_meshdesc(md, props):
    """ExportCollision from an imported collision MeshDesc (used by tests and round trips)."""
    kind = COLLISION_KINDS.get(str(props.get('nif_collision', 'MESH')), 'MESH')
    return ExportCollision(kind, np.asarray(md.vertices, dtype=np.float64), np.asarray(md.faces, dtype=np.int64), dict(props))


@dataclass
class ObjDesc:
    name: str
    kind: str                          # 'EMPTY' | 'MESH' | 'ARMATURE'
    matrix: np.ndarray                 # local to parent, Blender units
    parent: Optional['ObjDesc'] = None
    mesh: Optional[MeshDesc] = None
    bones: List[BoneDesc] = field(default_factory=list)
    armature: Optional['ObjDesc'] = None   # armature a skinned mesh is bound to
    xmesh: Optional['ExportMesh'] = None   # export only
    xcoll: Optional['ExportCollision'] = None   # export only (collision helpers)
    collection: str = 'main'               # 'main' | 'collision'
    display: Optional[str] = None          # 'WIRE'
    hidden: bool = False
    props: dict = field(default_factory=dict)


@dataclass
class SceneDesc:
    objects: List[ObjDesc] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------- helpers
def _v3(d):
    return np.array([d['x'], d['y'], d['z']], dtype=np.float64)


def rot33(r):
    """NiMatrix3 (dict m11..m33) -> 3x3 column-vector rotation matrix."""
    return np.array([[r['m11'], r['m21'], r['m31']],
                     [r['m12'], r['m22'], r['m32']],
                     [r['m13'], r['m23'], r['m33']]], dtype=np.float64)


def rot44(m):
    """Havok Matrix44 (bhkTransformShape): column-vector matrix, translation in m14/m24/m34 (Havok units)."""
    return np.array([[m['m11'], m['m12'], m['m13'], m['m14']],
                     [m['m21'], m['m22'], m['m23'], m['m24']],
                     [m['m31'], m['m32'], m['m33'], m['m34']],
                     [m['m41'], m['m42'], m['m43'], m['m44']]], dtype=np.float64)


def transform_matrix(t):
    """Block with Rotation/Translation/Scale (NiAVObject or NiTransform) -> 4x4 (NIF units)."""
    M = np.eye(4)
    M[:3, :3] = rot33(t['Rotation']) * float(t['Scale'])
    M[:3, 3] = _v3(t['Translation'])
    return M


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def scale_translation(M, s):
    M = M.copy()
    M[:3, 3] *= s
    return M


def strips_to_triangles(strips):
    tris = []
    for pts in strips:
        pts = np.asarray(pts, dtype=np.int64)
        if len(pts) < 3:
            continue
        a, b, c = pts[:-2], pts[1:-1], pts[2:]
        flip = (np.arange(len(a)) & 1).astype(bool)
        t = np.stack([a, np.where(flip, c, b), np.where(flip, b, c)], axis=1)
        keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
        tris.append(t[keep])
    return np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)


def bone_axis_matrix(fwd):
    """Columns = Blender bone axes (X, Y=forward, Z=up) expressed in the NIF node's local frame.
    ``fwd`` is 0..2 for +X,+Y,+Z and 3..5 for -X,-Y,-Z."""
    f = np.zeros(3)
    f[fwd % 3] = -1.0 if fwd >= 3 else 1.0
    u = np.zeros(3)
    u[(fwd + 1) % 3] = 1.0
    return np.stack([np.cross(f, u), f, u], axis=1)


def _norm_name(p):
    return p.replace('\\', '/').lower()


# ------------------------------------------------------- texture resolution
class TextureResolver:
    def __init__(self, nif_path, texture_dir):
        self.roots = []
        if texture_dir:
            self.roots.append(texture_dir)
        if nif_path:
            d = os.path.dirname(os.path.abspath(nif_path))
            self.roots.append(d)
            # walk up looking for a game "Data" tree (…/Data/meshes/x/y.nif -> …/Data)
            cur = d
            for _ in range(8):
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                if os.path.basename(cur).lower() in ('meshes', 'data'):
                    self.roots.append(parent if os.path.basename(cur).lower() == 'meshes' else cur)
                cur = parent
        self._dircache = {}
        self._cache = {}

    def _listdir(self, d):
        if d not in self._dircache:
            try:
                self._dircache[d] = {n.lower(): n for n in os.listdir(d)}
            except OSError:
                self._dircache[d] = {}
        return self._dircache[d]

    def _ci_join(self, root, rel):
        """Case-insensitive path join (game data is case-insensitive, Linux is not)."""
        cur = root
        for part in [p for p in rel.replace('\\', '/').split('/') if p]:
            names = self._listdir(cur)
            hit = names.get(part.lower())
            if hit is None:
                return None
            cur = os.path.join(cur, hit)
        return cur if os.path.isfile(cur) else None

    def resolve(self, nif_path):
        key = _norm_name(nif_path)
        if key in self._cache:
            return self._cache[key]
        found = None
        rel = nif_path.replace('\\', '/')
        variants = [rel]
        stem, ext = os.path.splitext(rel)
        for e in ('.dds', '.png', '.tga', '.bmp', '.jpg'):
            if e != ext.lower():
                variants.append(stem + e)
        for root in self.roots:
            for v in variants:
                found = self._ci_join(root, v)
                if found:
                    break
                if not v.lower().startswith('textures/'):
                    found = self._ci_join(root, 'textures/' + v)
                    if found:
                        break
            if found:
                break
        if not found:                      # last resort: bare file name next to the NIF
            base = os.path.basename(rel)
            for root in self.roots[-1:] if self.roots else []:
                found = self._ci_join(root, base)
        self._cache[key] = found
        return found

    def sibling(self, resolved, suffix):
        """<name>.ext -> <name><suffix>.ext (case-insensitive) next to ``resolved``."""
        if not resolved:
            return None
        d, fn = os.path.split(resolved)
        stem, ext = os.path.splitext(fn)
        names = self._listdir(d)
        hit = names.get((stem + suffix + ext).lower())
        return os.path.join(d, hit) if hit else None


# ------------------------------------------------------------ scene builder
TEX_SLOTS = [('Base Texture', 'base'), ('Dark Texture', 'dark'), ('Detail Texture', 'detail'),
             ('Gloss Texture', 'gloss'), ('Glow Texture', 'glow'), ('Bump Map Texture', 'bump'),
             ('Normal Texture', 'normal'), ('Parallax Texture', 'parallax'),
             ('Decal 0 Texture', 'decal0'), ('Decal 1 Texture', 'decal1'),
             ('Decal 2 Texture', 'decal2'), ('Decal 3 Texture', 'decal3')]


class SceneBuilder:
    def __init__(self, nif, settings: ImportSettings, nif_path: str = ''):
        self.nif = nif
        self.s = settings
        self.nif_path = nif_path
        self.scene = SceneDesc()
        self.warn = self.scene.warnings
        self.textures = TextureResolver(nif_path, settings.texture_dir)
        self._mat_cache = {}
        self.parent_of = {}           # block index -> parent NiNode block
        self.world = {}               # block index -> 4x4 world matrix (NIF units, nif root space)
        self.obj_of = {}              # block index -> ObjDesc
        self.bone_set = set()         # block indices of bone nodes
        self.rest = {}                # block index -> armature-space rest matrix (NIF units)
        self.armatures = {}           # skeleton root block index -> ObjDesc
        self.skinned = []             # (shape block, skin instance block)

    # ---- generic helpers -------------------------------------------------
    def d(self, ref):
        return self.nif.deref(None if ref is None or int(ref) < 0 else ref)

    def refs(self, arr):
        out = []
        for r in (arr if arr is not None else []):
            b = self.d(r)
            if b is not None:
                out.append(b)
        return out

    def is_node(self, b):
        return b.is_a('NiNode')

    def is_geom(self, b):
        return b.is_a('NiTriShape', 'NiTriStrips')

    # ---- main --------------------------------------------------------------
    def build(self):
        nif = self.nif
        for w in nif.warnings:
            self.warn.append(w)
        roots = [r for r in nif.roots if self.is_node(r) or self.is_geom(r)]
        if not roots:
            roots = [b for b in nif.blocks if self.is_node(b) or self.is_geom(b)][:1]
        if not roots:
            self.warn.append('No NiNode / geometry blocks found in file.')
            return self.scene
        for r in roots:
            self._index_tree(r, None, np.eye(4))
        self._find_skinning(roots)
        for r in roots:
            self._emit(r, None, {})
        if self.s.import_collision:
            for r in roots:
                self._emit_collision_tree(r)
        st = self.scene.stats
        st['objects'] = len(self.scene.objects)
        st['meshes'] = sum(1 for o in self.scene.objects if o.kind == 'MESH' and o.collection == 'main')
        st['armatures'] = len(self.armatures)
        st['bones'] = sum(len(o.bones) for o in self.armatures.values())
        return self.scene

    # ---- pass 1: hierarchy / world transforms ------------------------------
    def _index_tree(self, blk, parent, parent_world):
        if blk.index in self.world:
            return
        M = parent_world @ transform_matrix(blk)
        self.world[blk.index] = M
        if parent is not None:
            self.parent_of[blk.index] = parent
        if self.is_node(blk):
            for c in self.refs(blk['Children']):
                if self.is_node(c) or self.is_geom(c) or c.is_a('NiAVObject'):
                    self._index_tree(c, blk, M)

    def _children(self, blk):
        if not self.is_node(blk):
            return []
        return [c for c in self.refs(blk['Children']) if c.index in self.world]

    # ---- pass 2: skinning / bones ------------------------------------------
    def _find_skinning(self, roots):
        for idx in sorted(self.world):
            b = self.nif.blocks[idx]
            if self.is_geom(b):
                si = self.d(b['Skin Instance'])
                if si is not None and si.is_a('NiSkinInstance'):
                    self.skinned.append((b, si))
        if not self.s.import_skeleton:
            return
        skel_roots = {}
        for shape, si in self.skinned:
            sr = self.d(si['Skeleton Root'])
            bones = [self.d(r) for r in si['Bones']]
            bones = [x for x in bones if x is not None and x.index in self.world]
            if not bones:
                continue
            if sr is None or sr.index not in self.world:
                # fall back: lowest common ancestor of the bones
                sr = self._common_ancestor(bones)
            skel_roots.setdefault(sr.index, []).append((shape, si, bones))
        # skeleton-only NIFs (no skinned geometry): all NiNodes are bones
        if not skel_roots:
            root = roots[0]
            bsx = [e for e in self.refs(root['Extra Data List']) if e.btype == 'BSXFlags']
            is_skel = bool(bsx and (int(bsx[0]['Integer Data']) & 4))
            names = [self.nif.blocks[i]['Name'] for i in self.world if self.is_node(self.nif.blocks[i])]
            if is_skel or sum(1 for n in names if n.startswith('Bip01')) >= 3:
                skel_roots[root.index] = []
        for sr_idx, items in skel_roots.items():
            sr = self.nif.blocks[sr_idx]
            bones = set()
            for shape, si, blist in items:
                for bn in blist:
                    cur = bn
                    while cur is not None and cur.index != sr_idx:
                        bones.add(cur.index)
                        cur = self.parent_of.get(cur.index)
            if not items:                    # skeleton file
                for i in self.world:
                    b = self.nif.blocks[i]
                    if self.is_node(b) and i != sr_idx:
                        bones.add(i)
            # plain NiNode descendants of bones are bones too (leaf/helper bones)
            changed = True
            while changed:
                changed = False
                for i in list(self.world):
                    if i in bones or i == sr_idx:
                        continue
                    b = self.nif.blocks[i]
                    p = self.parent_of.get(i)
                    if p is not None and p.index in bones and self.is_node(b) and not self._has_geometry(b):
                        bones.add(i)
                        changed = True
            self.bone_set |= bones
            self._compute_rest(sr, items, bones)

    def _has_geometry(self, node):
        for c in self._children(node):
            if self.is_geom(c) or (self.is_node(c) and self._has_geometry(c)):
                return True
        return False

    def _common_ancestor(self, nodes):
        def chain(n):
            c = []
            while n is not None:
                c.append(n.index)
                n = self.parent_of.get(n.index)
            return c[::-1]
        chains = [chain(n) for n in nodes]
        common = None
        for i in range(min(len(c) for c in chains)):
            if all(c[i] == chains[0][i] for c in chains):
                common = chains[0][i]
            else:
                break
        return self.nif.blocks[common if common is not None else chains[0][0]]

    def _compute_rest(self, sr, items, bones):
        """Rest (bind) matrices in skeleton-root space (NIF units)."""
        inv_sr = np.linalg.inv(self.world[sr.index])
        bind = {}
        # prefer geometries influencing the most bones
        for shape, si, blist in sorted(items, key=lambda t: -len(t[2])):
            sd = self.d(si['Data'])
            if sd is None:
                continue
            G = inv_sr @ self.world[shape.index]
            for bi, bref in enumerate(si['Bones']):
                bn = self.d(bref)
                if bn is None or bi >= len(sd['Bone List']):
                    continue
                off = transform_matrix(sd['Bone List'][bi]['Skin Transform'])
                try:
                    cand = G @ np.linalg.inv(off)
                except np.linalg.LinAlgError:
                    continue
                if bn.index not in bind:
                    bind[bn.index] = cand
                elif np.abs(bind[bn.index] - cand).max() > 0.05 * max(1.0, np.abs(cand[:3, 3]).max()) * 0.1:
                    self.warn.append("Skinned shapes disagree about the bind pose of bone '%s' "
                                     "(max diff %.3f); using the first." %
                                     (bn['Name'], np.abs(bind[bn.index] - cand).max()))
        rest = {sr.index: np.eye(4)}

        def walk(node):
            for c in self._children(node):
                if c.index in bones:
                    if c.index in bind:
                        rest[c.index] = bind[c.index]
                    else:       # non skin-referenced bone: keep local transform to parent's rest
                        rest[c.index] = rest[node.index] @ transform_matrix(c)
                    walk(c)
                elif self.is_node(c) or self.is_geom(c):
                    rest[c.index] = rest[node.index] @ transform_matrix(c)
                    if self.is_node(c):
                        walk(c)
        walk(sr)
        self.rest.update(rest)
        self._bind_geom = getattr(self, '_bind_geom', {})
        self._skel_root_of = getattr(self, '_skel_root_of', {})
        for shape, si, blist in items:
            self._skel_root_of[shape.index] = sr.index
        self._pending_skeleton = getattr(self, '_pending_skeleton', [])
        self._pending_skeleton.append((sr, bones))

    # ---- pass 3: emit objects ----------------------------------------------
    def _emit(self, blk, parent_obj, ctx):
        """Create ObjDesc for ``blk`` and recurse.

        ``ctx`` carries 'armature' (ObjDesc) / 'arm_root' (skeleton root index) while
        inside a skeleton subtree.
        """
        s = self.s.scale
        idx = blk.index
        local = transform_matrix(blk)
        is_skel_root = any(sr.index == idx for sr, _ in getattr(self, '_pending_skeleton', []))

        if is_skel_root:
            bones_set = [b for sr, b in self._pending_skeleton if sr.index == idx][0]
            arm = ObjDesc(blk['Name'] or 'Armature', 'ARMATURE', scale_translation(local, s), parent_obj)
            arm.bones = self._make_bones(blk, bones_set)
            arm.props['nif_bone_forward'] = int(self._last_fwd)
            self.scene.objects.append(arm)
            self.armatures[idx] = arm
            self.obj_of[idx] = arm
            self._attach_props(arm, blk)
            for c in self._children(blk):
                self._emit(c, arm, {'armature': arm, 'arm_root': idx})
            return arm

        arm = ctx.get('armature')
        if idx in self.bone_set:
            # bones are part of the armature; shapes below them are handled by _emit of children
            for c in self._children(blk):
                self._emit(c, parent_obj, ctx)
            return None

        if self.is_geom(blk):
            return self._emit_geom(blk, parent_obj, ctx)

        if self.is_node(blk):
            if arm is not None and parent_obj is arm:
                parent_node = self.parent_of.get(idx)
                if parent_node is not None and parent_node.index == ctx['arm_root']:
                    M = local
                else:
                    M = self.rest.get(idx, local)
            else:
                M = local
            # ensure child matrices are relative to this empty
            if not self.s.import_empties and not self._is_meaningful(blk):
                # flatten: children get composed matrices
                for c in self._children(blk):
                    self._emit(c, parent_obj, dict(ctx, flatten=ctx.get('flatten', np.eye(4)) @ local))
                return None
            flat = ctx.get('flatten')
            if flat is not None:
                M = flat @ M
            obj = ObjDesc(blk['Name'] or 'NiNode', 'EMPTY', scale_translation(M, s), parent_obj)
            obj.hidden = bool(int(blk['Flags']) & 1)
            self.scene.objects.append(obj)
            self.obj_of[idx] = obj
            self._attach_props(obj, blk)
            ctx2 = {k: v for k, v in ctx.items() if k != 'flatten'}
            for c in self._children(blk):
                self._emit(c, obj, ctx2)
            return obj
        return None

    def _is_meaningful(self, node):
        return node.index in [r.index for r in self.nif.roots]

    def _attach_props(self, obj, blk):
        for e in self.refs(blk['Extra Data List']):
            if e.btype == 'BSXFlags':
                obj.props['nif_bsx_flags'] = int(e['Integer Data'])
            elif e.is_a('NiStringExtraData'):
                obj.props['nif_extra_' + (e['Name'] or 'string')] = e['String Data']
            elif e.is_a('NiIntegerExtraData'):
                obj.props['nif_extra_' + (e['Name'] or 'int')] = int(e['Integer Data'])
            elif e.is_a('NiFloatExtraData'):
                obj.props['nif_extra_' + (e['Name'] or 'float')] = float(e['Float Data'])
            elif e.btype == 'BSBound':
                obj.props['nif_bsbound'] = ' '.join('%.9g' % x for x in (*e['Center'], *e['Dimensions']))
        obj.props['nif_block'] = blk.btype

    # ---- bones ---------------------------------------------------------------
    def _make_bones(self, sr, bones):
        s = self.s.scale
        out = []
        order = []

        def walk(node, parent_bone):
            for c in self._children(node):
                if c.index in bones:
                    order.append((c, parent_bone))
                    walk(c, c)
                elif self.is_node(c):
                    walk(c, parent_bone)
        walk(sr, None)

        # forward axis: dominant axis of child offsets in the parent's local frame
        votes = np.zeros(6)
        for c, pb in order:
            if pb is None:
                continue
            t = transform_matrix(c)[:3, 3]
            if np.abs(t).max() < 1e-4:
                continue
            ax = int(np.argmax(np.abs(t)))
            votes[ax + (3 if t[ax] < 0 else 0)] += 1
        fwd = int(np.argmax(votes)) if votes.max() > 0 else 0       # default +X (Bethesda skeletons)
        C = bone_axis_matrix(fwd)
        self._last_fwd = fwd

        heads = {c.index: self.rest[c.index][:3, 3] for c, _ in order}
        children = {}
        for c, pb in order:
            if pb is not None:
                children.setdefault(pb.index, []).append(c.index)
        default_len = 4.0
        for c, pb in order:
            R = self.rest[c.index].copy()
            # normalise scale out of the rest matrix
            sc = np.cbrt(abs(np.linalg.det(R[:3, :3]))) or 1.0
            R3 = R[:3, :3] / sc
            M = np.eye(4)
            M[:3, :3] = R3 @ C
            M[:3, 3] = R[:3, 3] * s
            kids = children.get(c.index, [])
            if kids:
                dists = [np.linalg.norm(heads[k] - heads[c.index]) for k in kids]
                dists = [x for x in dists if x > 1e-3]
                length = (min(dists) if dists else default_len) * s
            else:
                length = default_len * s
            out.append(BoneDesc(c['Name'] or 'Bone', pb['Name'] if pb is not None else None,
                                M, max(length, 1e-3), R))
        return out

    # ---- geometry --------------------------------------------------------------
    def _inherited_props(self, blk):
        """Properties on the shape plus those inherited from ancestors (nearest wins)."""
        found = {}
        chain = []
        cur = blk
        while cur is not None:
            chain.append(cur)
            cur = self.parent_of.get(cur.index)
        for node in chain:                       # nearest first
            if not node.is_a('NiAVObject'):
                continue
            for p in self.refs(node['Properties']):
                found.setdefault(p.btype if not p.is_a('NiProperty') else self._prop_class(p), p)
        return found

    @staticmethod
    def _prop_class(p):
        for c in ('NiMaterialProperty', 'NiTexturingProperty', 'NiAlphaProperty', 'NiStencilProperty',
                  'NiSpecularProperty', 'NiVertexColorProperty', 'NiZBufferProperty',
                  'NiWireframeProperty', 'NiShadeProperty', 'NiDitherProperty', 'NiFogProperty'):
            if p.is_a(c):
                return c
        return p.btype

    def _make_material(self, blk):
        props = self._inherited_props(blk)
        key = tuple(sorted((k, v.index) for k, v in props.items()))
        if not key:
            return None
        if key in self._mat_cache:
            return self._mat_cache[key]
        mp = props.get('NiMaterialProperty')
        name = (mp['Name'] if mp is not None and mp['Name'] else None) or (blk['Name'] or 'Material')
        m = MaterialDesc(key=key, name=name)
        if mp is not None:
            m.diffuse = tuple(float(mp['Diffuse Color'][c]) for c in 'rgb')
            m.ambient = tuple(float(mp['Ambient Color'][c]) for c in 'rgb')
            m.specular = tuple(float(mp['Specular Color'][c]) for c in 'rgb')
            m.emissive = tuple(float(mp['Emissive Color'][c]) for c in 'rgb')
            m.glossiness = float(mp['Glossiness'])
            m.alpha = float(mp['Alpha'])
        sp = props.get('NiSpecularProperty')
        m.specular_enabled = bool(sp is not None and (int(sp['Flags']) & 1))
        tp = props.get('NiTexturingProperty')
        if tp is not None:
            for field_name, slot in TEX_SLOTS:
                desc = tp.get(field_name)
                if not desc or not tp.get('Has ' + field_name, True):
                    continue
                src = self.d(desc.get('Source'))
                if src is None or not src.is_a('NiSourceTexture'):
                    continue
                path = src['File Name'] if int(src['Use External']) == 1 else ''
                if not path:
                    self.warn.append('Texture %s of %s is embedded/empty; skipped.' % (slot, name))
                    continue
                ts = TextureSlot(slot, path, self.textures.resolve(path), int(desc.get('UV Set', 0)),
                                 int(desc.get('Clamp Mode', 3)), int(desc.get('Filter Mode', 2)))
                if desc.get('Has Texture Transform'):
                    ts.transform = dict(translation=(desc['Translation']['u'], desc['Translation']['v']),
                                        scale=(desc['Scale']['u'], desc['Scale']['v']),
                                        rotation=float(desc['Rotation']))
                m.textures[slot] = ts
            base = m.textures.get('base')
            if base is not None and self.s.find_normal_maps and 'normal' not in m.textures:
                n = self.textures.sibling(base.resolved, '_n')
                if n:
                    npath = os.path.splitext(base.path)[0] + '_n' + os.path.splitext(base.path)[1]
                    m.textures['normal'] = TextureSlot('normal', npath, n)
        ap = props.get('NiAlphaProperty')
        if ap is not None:
            fl = int(ap['Flags'])
            m.alpha_blend = bool(fl & 1)
            m.alpha_test = bool(fl & 0x200)
            m.alpha_src = (fl >> 1) & 15
            m.alpha_dst = (fl >> 5) & 15
            m.alpha_threshold = int(ap['Threshold']) / 255.0
        st = props.get('NiStencilProperty')
        if st is not None:
            m.two_sided = int(st.get('Draw Mode', 1)) == 3 or bool((int(st.get('Flags', 0)) >> 10) & 3 == 3)
        m.props = {'nif_material_block': mp.index if mp is not None else -1}
        self._mat_cache[key] = m
        return m

    def _emit_geom(self, blk, parent_obj, ctx):
        s = self.s.scale
        data = self.d(blk['Data'])
        if data is None or not data.is_a('NiTriBasedGeomData'):
            self.warn.append("Shape '%s' has no triangle data; skipped." % blk['Name'])
            return None
        verts = np.asarray(data['Vertices'], dtype=np.float64).reshape(-1, 3) if data.get('Has Vertices') else None
        if verts is None or len(verts) == 0:
            return None
        n = len(verts)
        if data.is_a('NiTriShapeData'):
            tris = np.asarray(data['Triangles'], dtype=np.int64).reshape(-1, 3)
        else:
            tris = strips_to_triangles(data['Points']) if data.get('Has Points', True) else np.zeros((0, 3), np.int64)
        # drop degenerate / out-of-range triangles
        ok = (tris < n).all(axis=1) & (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
        if not ok.all():
            self.warn.append("Shape '%s': dropped %d degenerate triangles." % (blk['Name'], int((~ok).sum())))
            tris = tris[ok]
        normals = None
        if self.s.import_normals and data.get('Has Normals'):
            normals = np.asarray(data['Normals'], dtype=np.float64).reshape(-1, 3)
        uvs = []
        uv_arr = data.get('UV Sets')
        if uv_arr is not None and len(uv_arr):
            for k in range(len(uv_arr)):
                uv = np.array(uv_arr[k], dtype=np.float64).reshape(-1, 2)
                uv[:, 1] = 1.0 - uv[:, 1]
                uvs.append(uv)
        colors = None
        if self.s.import_vertex_colors and data.get('Has Vertex Colors'):
            colors = np.asarray(data['Vertex Colors'], dtype=np.float64).reshape(-1, 4)

        skin = None
        si = self.d(blk['Skin Instance'])
        arm = None
        if si is not None and si.is_a('NiSkinInstance') and self.s.import_skeleton:
            sr_idx = getattr(self, '_skel_root_of', {}).get(blk.index)
            arm = self.armatures.get(sr_idx) if sr_idx is not None else None
            if arm is not None:
                skin = self._skin_weights(blk, si, n)
                if skin is None or not skin.weights:
                    arm = None
                    skin = None
        mesh = MeshDesc(blk['Name'] or 'Shape', verts * s, tris.astype(np.int32),
                        normals, uvs, colors, self._make_material(blk), skin)

        if arm is not None:
            sr = self.nif.blocks[self._skel_root_of[blk.index]]
            G = np.linalg.inv(self.world[sr.index]) @ self.world[blk.index]
            M = scale_translation(G, s)
            obj = ObjDesc(blk['Name'] or 'Shape', 'MESH', M, arm, mesh, armature=arm)
        else:
            if arm is None and ctx.get('armature') is not None and parent_obj is ctx['armature']:
                pn = self.parent_of.get(blk.index)
                M = transform_matrix(blk) if (pn is not None and pn.index == ctx['arm_root']) \
                    else self.rest.get(blk.index, transform_matrix(blk))
            else:
                M = transform_matrix(blk)
            flat = ctx.get('flatten')
            if flat is not None:
                M = flat @ M
            obj = ObjDesc(blk['Name'] or 'Shape', 'MESH', scale_translation(M, s), parent_obj, mesh)
        obj.hidden = bool(int(blk['Flags']) & 1)
        if obj.hidden and not self.s.import_hidden:
            return None
        self.scene.objects.append(obj)
        self.obj_of[blk.index] = obj
        self._attach_props(obj, blk)
        return obj

    def _skin_weights(self, blk, si, nverts):
        sd = self.d(si['Data'])
        if sd is None:
            return None
        bones = [self.d(r) for r in si['Bones']]
        skin = SkinDesc()
        if sd.get('Has Vertex Weights', True) and len(sd['Bone List']) and \
                any(len(b['Vertex Weights']) for b in sd['Bone List']):
            for bi, bd in enumerate(sd['Bone List']):
                bn = bones[bi] if bi < len(bones) else None
                if bn is None:
                    continue
                vw = bd['Vertex Weights']
                if len(vw) == 0:
                    continue
                idx = np.asarray(vw['Index'], dtype=np.int64)
                w = np.asarray(vw['Weight'], dtype=np.float64)
                keep = (idx < nverts) & (w > 0)
                if not keep.any():
                    continue
                name = bn['Name'] or 'Bone'
                if name in skin.weights:        # duplicated bone names: merge
                    i0, w0 = skin.weights[name]
                    skin.weights[name] = (np.concatenate([i0, idx[keep]]), np.concatenate([w0, w[keep]]))
                else:
                    skin.weights[name] = (idx[keep], w[keep])
            return skin
        # fallback: weights stored only in the skin partition
        sp = self.d(si.get('Skin Partition'))
        if sp is None:
            return None
        for part in sp['Partitions']:
            pb = part['Bones']
            vmap = np.asarray(part['Vertex Map']) if part.get('Has Vertex Map') else np.arange(part['Num Vertices'])
            W = np.asarray(part['Vertex Weights']) if part.get('Has Vertex Weights') else None
            B = np.asarray(part['Bone Indices']) if part.get('Has Bone Indices') else None
            if W is None or B is None:
                continue
            for k in range(W.shape[1]):
                for local_bone in np.unique(B[:, k]):
                    sel = (B[:, k] == local_bone) & (W[:, k] > 0)
                    if not sel.any() or local_bone >= len(pb):
                        continue
                    bn = bones[int(pb[local_bone])]
                    if bn is None:
                        continue
                    name = bn['Name'] or 'Bone'
                    i1, w1 = vmap[sel].astype(np.int64), W[sel, k].astype(np.float64)
                    if name in skin.weights:
                        i0, w0 = skin.weights[name]
                        skin.weights[name] = (np.concatenate([i0, i1]), np.concatenate([w0, w1]))
                    else:
                        skin.weights[name] = (i1, w1)
        return skin

    # ---- collision ----------------------------------------------------------------
    def _emit_collision_tree(self, blk):
        co = self.d(blk['Collision Object']) if blk.is_a('NiAVObject') else None
        if co is not None:
            try:
                self._emit_collision(blk, co)
            except Exception as e:                                  # noqa: BLE001
                self.warn.append('Collision of %s could not be imported: %s' % (blk['Name'], e))
        for c in self._children(blk):
            self._emit_collision_tree(c)

    def _emit_collision(self, node_blk, co):
        body = self.d(co.get('Body'))
        if body is None:
            return
        shape = self.d(body.get('Shape'))
        info = body.get('Rigid Body Info') or {}
        M = np.eye(4)
        if info and body.btype in ('bhkRigidBodyT', 'bhkRigidBody'):
            q = info.get('Rotation')
            t = info.get('Translation')
            if q is not None and t is not None:
                M[:3, :3] = quat_to_matrix(q['x'], q['y'], q['z'], q['w'])
                M[:3, 3] = np.array([t['x'], t['y'], t['z']]) * HAVOK_SCALE
        parent = self.obj_of.get(node_blk.index)
        # object matrix is relative to the node object; for bone nodes use the armature
        if parent is None:
            arm = next(iter(self.armatures.values()), None)
            parent = arm
            if node_blk.index in self.rest:
                M = self.rest[node_blk.index] @ M
        s = self.s.scale
        hk_props = self._havok_props(co, body, shape)
        for k, (name, verts, faces, hull) in enumerate(self._shape_meshes(shape, np.eye(4))):
            if len(verts) == 0:
                continue
            md = MeshDesc('%s_collision%s' % (node_blk['Name'] or 'node', '' if k == 0 else k),
                          verts * s, faces.astype(np.int32), convex_hull=hull)
            obj = ObjDesc(md.name, 'MESH', scale_translation(M, s), parent, md, collection='collision', display='WIRE')
            obj.props['nif_collision'] = name
            obj.props.update(hk_props)
            self.scene.objects.append(obj)

    def _first_havok_material(self, shape):
        for _ in range(8):
            if shape is None:
                return None
            bt = shape.btype
            if bt == 'bhkPackedNiTriStripsShape':
                subs = shape.get('Sub Shapes')
                return subs[0]['Material'] if subs is not None and len(subs) else None
            if bt == 'bhkListShape':
                subs = shape['Sub Shapes']
                if len(subs) == 0:
                    return shape.get('Material')
                shape = self.d(subs[0])
            elif bt in ('bhkMoppBvTreeShape', 'bhkTransformShape', 'bhkConvexTransformShape'):
                shape = self.d(shape['Shape'])
            else:
                return shape.get('Material')
        return None

    def _havok_props(self, co, body, shape):
        """Rigid body / shape settings (+ the MOPP of an untouched packed mesh) stored as nif_* custom props."""
        import base64
        import hashlib
        p = {}
        try:
            info = body.get('Rigid Body Info') or {}
            hf = body.get('Havok Filter') or {}
            p['nif_hk_layer'] = int(hf.get('Layer', 1))
            p['nif_hk_col_flags'] = int(co.get('Flags', 1))
            for key, name in (('Mass', 'mass'), ('Friction', 'friction'), ('Restitution', 'restitution'),
                              ('Linear Damping', 'linear_damping'), ('Angular Damping', 'angular_damping')):
                if key in info:
                    p['nif_hk_' + name] = float(info[key])
            for key, name in (('Motion System', 'motion'), ('Quality Type', 'quality'),
                              ('Deactivator Type', 'deactivator'), ('Solver Deactivation', 'solver_deactivation')):
                if key in info:
                    p['nif_hk_' + name] = int(info[key])
            p['nif_hk_body'] = body.btype
            mat = self._first_havok_material(shape)
            if mat is not None:
                p['nif_hk_material'] = int(mat['Material'] if hasattr(mat, 'items') else mat)
            if shape is not None and shape.btype == 'bhkMoppBvTreeShape':
                packed = self.d(shape['Shape'])
                if packed is not None and packed.btype == 'bhkPackedNiTriStripsShape':
                    data = self.d(packed['Data'])
                    code = shape['MOPP Code']
                    if data is not None and len(code['Data']):
                        tr = data['Triangles']
                        if isinstance(tr, np.ndarray) and tr.dtype.names:
                            f = np.stack([tr['Triangle']['v1'], tr['Triangle']['v2'], tr['Triangle']['v3']], 1)
                            weld = tr['Welding Info']
                        else:
                            f = np.array([[t['Triangle']['v1'], t['Triangle']['v2'], t['Triangle']['v3']] for t in tr])
                            weld = np.array([t['Welding Info'] for t in tr])
                        f = np.asarray(f, dtype=np.int32).reshape(-1, 3)
                        sc = packed.get('Scale')
                        v = np.asarray(data['Vertices'], dtype=np.float32).reshape(-1, 3)
                        if sc:
                            v = (v * np.array([sc['x'], sc['y'], sc['z']])).astype(np.float32)
                        off = code['Offset']
                        p['nif_mopp_data'] = base64.b64encode(bytes(code['Data'])).decode('ascii')
                        p['nif_mopp_offset'] = ' '.join('%.9g' % off[k] for k in ('x', 'y', 'z', 'w'))
                        p['nif_mopp_scale'] = float(shape['Scale'])
                        p['nif_mopp_unused'] = bytes(shape['Unused 01']).hex()
                        p['nif_mopp_verts'] = base64.b64encode(v.tobytes()).decode('ascii')
                        p['nif_mopp_faces'] = hashlib.sha1(f.tobytes()).hexdigest()
                        p['nif_mopp_weld'] = base64.b64encode(np.asarray(weld, dtype='<u2').tobytes()).decode('ascii')
        except Exception as e:                                              # noqa: BLE001
            self.warn.append('Some Havok properties could not be read (%s)' % e)
        return p

    def _shape_meshes(self, shape, T):
        """Yield (type name, verts (NIF units), faces, convex_hull) for a Havok shape tree."""
        if shape is None:
            return
        bt = shape.btype
        if shape.is_a('bhkMoppBvTreeShape'):
            yield from self._shape_meshes(self.d(shape['Shape']), T)
        elif bt == 'bhkPackedNiTriStripsShape':
            data = self.d(shape['Data'])
            if data is None:
                return
            v = np.asarray(data['Vertices'], dtype=np.float64).reshape(-1, 3)
            sc = shape.get('Scale')
            if sc:
                v = v * np.array([sc['x'], sc['y'], sc['z']])
            tr = data['Triangles']
            if isinstance(tr, np.ndarray) and tr.dtype.names:
                f = np.stack([tr['Triangle']['v1'], tr['Triangle']['v2'], tr['Triangle']['v3']], axis=1)
            else:
                f = np.array([[t['Triangle']['v1'], t['Triangle']['v2'], t['Triangle']['v3']] for t in tr],
                             dtype=np.int64).reshape(-1, 3)
            yield bt, self._xf(v * HAVOK_SCALE, T), f.astype(np.int64), False
        elif bt == 'bhkNiTriStripsShape':
            for r in shape['Strips Data']:
                d = self.d(r)
                if d is None or not d.is_a('NiTriStripsData'):
                    continue
                v = np.asarray(d['Vertices'], dtype=np.float64).reshape(-1, 3)
                yield bt, self._xf(v * HAVOK_SCALE, T), strips_to_triangles(d['Points']), False
        elif bt == 'bhkListShape':
            for r in shape['Sub Shapes']:
                yield from self._shape_meshes(self.d(r), T)
        elif bt in ('bhkTransformShape', 'bhkConvexTransformShape'):
            A = rot44(shape['Transform'])
            A[:3, 3] *= HAVOK_SCALE
            yield from self._shape_meshes(self.d(shape['Shape']), T @ A)
        elif bt == 'bhkBoxShape':
            h = _v3(shape['Dimensions']) * HAVOK_SCALE
            v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64) * h
            yield bt, self._xf(v, T), np.zeros((0, 3), np.int64), True
        elif bt == 'bhkSphereShape':
            yield (bt,) + _uv_sphere(float(shape['Radius']) * HAVOK_SCALE, np.zeros(3), T)
        elif bt == 'bhkMultiSphereShape':
            for sp in shape['Spheres']:
                yield (bt,) + _uv_sphere(float(sp['Radius']) * HAVOK_SCALE, _v3(sp['Center']) * HAVOK_SCALE, T)
        elif bt == 'bhkCapsuleShape':
            p1 = _v3(shape['First Point']) * HAVOK_SCALE
            p2 = _v3(shape['Second Point']) * HAVOK_SCALE
            r = float(shape['Radius']) * HAVOK_SCALE
            va, _ = _uv_sphere(r, p1, T)[:2]
            vb, _ = _uv_sphere(r, p2, T)[:2]
            yield bt, np.concatenate([va, vb]), np.zeros((0, 3), np.int64), True
        elif bt == 'bhkConvexVerticesShape':
            v = np.asarray(shape['Vertices'], dtype=np.float64).reshape(-1, 4)[:, :3] * HAVOK_SCALE
            yield bt, self._xf(v, T), np.zeros((0, 3), np.int64), True
        else:
            self.warn.append('Collision shape %s is not supported; skipped.' % bt)

    @staticmethod
    def _xf(v, T):
        return v @ T[:3, :3].T + T[:3, 3]


def _uv_sphere(radius, center, T, rings=8, segs=12):
    verts = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segs):
            th = 2 * math.pi * j / segs
            verts.append([radius * math.sin(phi) * math.cos(th), radius * math.sin(phi) * math.sin(th),
                          radius * math.cos(phi)])
    v = np.array(verts) + center
    faces = []
    for i in range(rings):
        for j in range(segs):
            a = i * segs + j
            b = i * segs + (j + 1) % segs
            c = (i + 1) * segs + j
            d = (i + 1) * segs + (j + 1) % segs
            faces.append([a, c, b])
            faces.append([b, c, d])
    f = np.array(faces, dtype=np.int64)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return SceneBuilder._xf(v, T), f[ok], False


def build_scene(nif, settings=None, nif_path=''):
    return SceneBuilder(nif, settings or ImportSettings(), nif_path).build()
