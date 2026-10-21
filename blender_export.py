# SPDX-License-Identifier: GPL-3.0-or-later
"""Reads Blender objects into :class:`nif_scene.ObjDesc` for :mod:`nif_export` (Blender 4.0 API)."""
import os

import bpy
import numpy as np
from mathutils import Matrix

from . import nif_scene as SC

EXPORT_TYPES = {'MESH', 'EMPTY', 'ARMATURE'}


# ----------------------------------------------------------------- selection
def gather_objects(context, selected_only, include_hidden=False):
    if selected_only:
        objs = list(context.selected_objects)
    else:
        objs = list(context.scene.objects)
    objs = [o for o in objs if o.type in EXPORT_TYPES]
    if not include_hidden:
        objs = [o for o in objs if o.visible_get() or o in context.selected_objects]
    # skip imported collision helpers
    objs = [o for o in objs if 'nif_collision' not in o and o.users_collection and
            o.users_collection[0].name != 'Collision']
    chosen = set(objs)
    # pull in armatures (and their parents) needed by skinned meshes
    for o in list(objs):
        for arm in _armatures_of(o):
            cur = arm
            while cur is not None and cur not in chosen:
                if cur.type in EXPORT_TYPES:
                    chosen.add(cur)
                cur = cur.parent
    return [o for o in context.scene.objects if o in chosen]


def is_collision_helper(o):
    return o.type == 'MESH' and ('nif_collision' in o or (o.users_collection and o.users_collection[0].name == 'Collision'))


def gather_collision(context, selected_only, chosen):
    """Collision helper meshes belonging to the export: children of exported objects, or selected / unparented ones."""
    sel = set(context.selected_objects)
    out = []
    for o in context.scene.objects:
        if not is_collision_helper(o):
            continue
        if o.parent is not None and o.parent in chosen:
            out.append(o)
        elif not selected_only or o in sel:
            out.append(o)
    return out


def _armatures_of(obj):
    if obj.type != 'MESH':
        return []
    return [m.object for m in obj.modifiers if m.type == 'ARMATURE' and m.object is not None]


# ------------------------------------------------------------------ materials
def _linked_node(sock):
    return sock.links[0].from_node if sock.is_linked else None


def _image_from(sock, through_mix=True):
    """Follow ``sock`` back to an Image Texture node (optionally through Mix / NormalMap nodes)."""
    node = _linked_node(sock)
    seen = 0
    while node is not None and seen < 8:
        seen += 1
        if node.bl_idname == 'ShaderNodeTexImage':
            return node
        if node.bl_idname == 'ShaderNodeNormalMap':
            node = _linked_node(node.inputs['Color'])
        elif node.bl_idname == 'ShaderNodeMix' and through_mix:
            nxt = _linked_node(node.inputs[6])
            node = nxt if nxt is not None else _linked_node(node.inputs[7])
        else:
            return None
    return None


def _tex_slot(slot, node):
    img = node.image
    if img is None:
        return None
    path = ''
    try:
        path = img.filepath_from_user()
    except Exception:
        path = bpy.path.abspath(img.filepath) if img.filepath else ''
    if not path:
        path = img.name
    ts = SC.TextureSlot(slot, path)
    ts.clamp_mode = 3 if node.extension == 'REPEAT' else 0
    return ts


def material_desc(mat, cache):
    if mat is None:
        return None
    if mat.name in cache:
        return cache[mat.name]
    md = SC.MaterialDesc(key=(mat.name,), name=mat.name)
    bsdf = None
    if mat.use_nodes and mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.bl_idname == 'ShaderNodeBsdfPrincipled':
                bsdf = n
                break
    if bsdf is not None:
        bc = bsdf.inputs['Base Color']
        base = _image_from(bc)
        if base is not None:
            ts = _tex_slot('base', base)
            if ts:
                md.textures['base'] = ts
            n = _linked_node(bc)
            if n is not None and n.bl_idname == 'ShaderNodeMix':       # multiply tint node
                md.diffuse = tuple(float(c) for c in n.inputs[7].default_value[:3])
        else:
            md.diffuse = tuple(float(c) for c in bc.default_value[:3])
        em = bsdf.inputs['Emission Color']
        glow = _image_from(em)
        if glow is not None:
            ts = _tex_slot('glow', glow)
            if ts:
                md.textures['glow'] = ts
        elif bsdf.inputs['Emission Strength'].default_value > 0:
            md.emissive = tuple(float(c) for c in em.default_value[:3])
        sp = bsdf.inputs['Specular IOR Level']
        gloss = _image_from(sp)
        if gloss is not None:
            ts = _tex_slot('gloss', gloss)
            if ts:
                md.textures['gloss'] = ts
            md.specular = (1.0, 1.0, 1.0)
            md.specular_enabled = True
        else:
            s = min(float(sp.default_value) * 2.0, 1.0)
            md.specular = (s, s, s)
            md.specular_enabled = s > 0.01
        nrm = _image_from(bsdf.inputs['Normal'])
        if nrm is not None:
            ts = _tex_slot('normal', nrm)
            if ts:
                md.textures['normal'] = ts
        rough = max(float(bsdf.inputs['Roughness'].default_value), 0.05)
        md.glossiness = min(max(2.0 / (rough * rough) - 2.0, 1.0), 128.0)
        if not bsdf.inputs['Alpha'].is_linked:
            md.alpha = float(bsdf.inputs['Alpha'].default_value)
        # labelled extra textures created by the importer
        for n in mat.node_tree.nodes:
            if n.bl_idname == 'ShaderNodeTexImage' and n.label in ('dark', 'detail', 'decal0') \
                    and n.label not in md.textures:
                ts = _tex_slot(n.label, n)
                if ts:
                    md.textures[n.label] = ts
    if 'nif_glossiness' in mat:
        md.glossiness = float(mat['nif_glossiness'])
    # transparency
    blend = getattr(mat, 'blend_method', 'OPAQUE')
    if getattr(mat, 'surface_render_method', None) == 'BLENDED':
        blend = 'BLEND'
    if 'nif_alpha_blend' in mat or 'nif_alpha_test' in mat:
        md.alpha_blend = bool(mat.get('nif_alpha_blend', False)) and blend != 'OPAQUE'
        md.alpha_test = bool(mat.get('nif_alpha_test', False))
    else:
        md.alpha_blend = blend == 'BLEND'
        md.alpha_test = blend in ('CLIP', 'HASHED')
    if blend == 'CLIP':
        md.alpha_threshold = float(getattr(mat, 'alpha_threshold', 0.5))
    md.two_sided = not mat.use_backface_culling
    cache[mat.name] = md
    return md


# --------------------------------------------------------------------- meshes
def _loop_normals(me):
    n = len(me.loops)
    arr = np.zeros(n * 3, dtype=np.float32)
    try:
        if hasattr(me, 'corner_normals'):
            me.corner_normals.foreach_get('vector', arr)
        else:
            raise AttributeError
    except Exception:
        me.calc_normals_split()
        me.loops.foreach_get('normal', arr)
    return arr.reshape(-1, 3)


def _read_colors(me, tri_loops, tri_verts):
    attr = me.color_attributes.active_color if hasattr(me.color_attributes, 'active_color') else None
    if attr is None and len(me.color_attributes):
        attr = me.color_attributes[0]
    if attr is None:
        return None
    data = np.zeros(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get('color', data)
    data = data.reshape(-1, 4)
    if attr.domain == 'CORNER':
        return data[tri_loops]
    if attr.domain == 'POINT':
        return data[tri_verts]
    return None


def mesh_to_xmeshes(obj, context, settings, arm_obj, mat_cache, sc):
    """Return list of ExportMesh (one per material slot used), geometry baked with the object's scale."""
    dg = context.evaluated_depsgraph_get()
    old_pose = None
    if arm_obj is not None:
        old_pose = arm_obj.data.pose_position
        arm_obj.data.pose_position = 'REST'
        context.view_layer.update()
        dg = context.evaluated_depsgraph_get()
    try:
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
    finally:
        if arm_obj is not None:
            arm_obj.data.pose_position = old_pose
    try:
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        if nt == 0:
            return []
        nv = len(me.vertices)
        co = np.zeros(nv * 3, dtype=np.float32)
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3).astype(np.float64)
        loops = np.zeros(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get('loops', loops)
        tverts = np.zeros(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get('vertices', tverts)
        midx = np.zeros(nt, dtype=np.int32)
        me.loop_triangles.foreach_get('material_index', midx)

        # bake object scale into the vertices (NIF only has uniform scale per node)
        scv = np.array([sc.x, sc.y, sc.z], dtype=np.float64)
        co = co * scv
        mirrored = scv.prod() < 0

        normals = None
        if settings['normals']:
            ln = _loop_normals(me).astype(np.float64)[loops]
            normals = ln / scv
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        uvs = []
        for layer in list(me.uv_layers)[:2]:
            a = np.zeros(len(me.loops) * 2, dtype=np.float32)
            layer.data.foreach_get('uv', a)
            uvs.append(a.reshape(-1, 2)[loops].astype(np.float64))
        colors = _read_colors(me, loops, tverts) if settings['colors'] else None
        if colors is not None:
            colors = colors.astype(np.float64)

        weights = None
        if arm_obj is not None:
            names = {g.index: g.name for g in ev.vertex_groups}
            bone_names = {b.name for b in arm_obj.data.bones}
            acc = {}
            for v in me.vertices:
                for g in v.groups:
                    nm = names.get(g.group)
                    if nm in bone_names and g.weight > 0:
                        acc.setdefault(nm, ([], []))
                        acc[nm][0].append(v.index)
                        acc[nm][1].append(g.weight)
            weights = {k: (np.array(i, dtype=np.int64), np.array(w, dtype=np.float64)) for k, (i, w) in acc.items()}

        out = []
        slots = list(obj.material_slots)
        used = sorted(set(midx.tolist()))
        for mi in used:
            sel = np.nonzero(midx == mi)[0]
            corner = (sel[:, None] * 3 + np.arange(3)).reshape(-1)
            if mirrored:
                corner = corner.reshape(-1, 3)[:, ::-1].reshape(-1)
            mat = slots[mi].material if mi < len(slots) else None
            name = obj.name if len(used) == 1 else '%s:%s' % (obj.name, mat.name if mat else mi)
            out.append(SC.ExportMesh(
                name, co, tverts[corner].astype(np.int64),
                normals[corner] if normals is not None else None,
                [u[corner] for u in uvs], colors[corner] if colors is not None else None,
                material_desc(mat, mat_cache), weights))
        return out
    finally:
        ev.to_mesh_clear()


# ------------------------------------------------------------------ scene build
def _mat(m):
    return np.array([list(r) for r in m], dtype=np.float64)


def collect(context, settings):
    """Blender objects -> list of ObjDesc (export side)."""
    objs = gather_objects(context, settings['selected_only'], settings.get('hidden', False))
    chosen = set(objs)
    descs = {}
    order = []
    mat_cache = {}
    # first create ObjDescs for armatures/empties/meshes
    for o in objs:
        if o.type == 'ARMATURE':
            d = SC.ObjDesc(o.name, 'ARMATURE', np.eye(4))
            d.bones = [SC.BoneDesc(b.name, b.parent.name if b.parent else None, _mat(b.matrix_local), b.length)
                       for b in o.data.bones]
            d.props = {k: o[k] for k in o.keys() if k.startswith('nif_') and isinstance(o[k], (int, float, str))}
            d.props.setdefault('nif_bone_forward', 0)
        elif o.type == 'EMPTY':
            d = SC.ObjDesc(o.name, 'EMPTY', np.eye(4))
            d.props = {k: o[k] for k in o.keys() if k.startswith('nif_') and isinstance(o[k], (int, float, str))}
        else:
            d = SC.ObjDesc(o.name, 'MESH', np.eye(4))
            d.props = {k: o[k] for k in o.keys() if k.startswith('nif_') and isinstance(o[k], (int, float, str))}
        descs[o] = d
        order.append((o, d))
    result = []
    for o, d in order:
        if o.type != 'MESH':
            parent = o.parent if o.parent in chosen else None
            d.parent = descs.get(parent)
            rel = (parent.matrix_world.inverted() @ o.matrix_world) if parent else o.matrix_world
            d.matrix = _mat(rel)
            result.append(d)
            continue
        arm_obj = next((a for a in _armatures_of(o) if a in chosen), None)
        parent = arm_obj if arm_obj is not None else (o.parent if o.parent in chosen else None)
        rel = (parent.matrix_world.inverted() @ o.matrix_world) if parent else o.matrix_world
        loc, rot, sc = rel.decompose()
        M = _mat(Matrix.LocRotScale(loc, rot, None))      # scale is baked into the vertices instead
        xs = mesh_to_xmeshes(o, context, settings, arm_obj, mat_cache, sc)
        for k, xm in enumerate(xs):
            dd = d if k == 0 else SC.ObjDesc(xm.name, 'MESH', np.eye(4), props=dict(d.props))
            dd.name = xm.name
            dd.xmesh = xm
            dd.matrix = M
            dd.parent = descs.get(parent)
            dd.armature = descs.get(arm_obj) if arm_obj is not None else None
            result.append(dd)
    if settings.get('collision', True):
        for o in gather_collision(context, settings['selected_only'], chosen):
            me = o.data
            if len(me.vertices) == 0:
                continue
            parent = o.parent if o.parent in chosen else None
            rel = (parent.matrix_world.inverted() @ o.matrix_world) if parent else o.matrix_world
            loc, rot, sc = rel.decompose()
            M = _mat(Matrix.LocRotScale(loc, rot, None))
            co = np.empty(len(me.vertices) * 3, dtype=np.float64)
            me.vertices.foreach_get('co', co)
            verts = co.reshape(-1, 3) * np.array([sc.x, sc.y, sc.z])
            me.calc_loop_triangles()
            tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
            me.loop_triangles.foreach_get('vertices', tri)
            props = {k: o[k] for k in o.keys() if k.startswith('nif_') and isinstance(o[k], (int, float, str))}
            d = SC.ObjDesc(o.name, 'MESH', M, descs.get(parent), collection='collision')
            d.props = props
            d.xcoll = SC.ExportCollision(SC.COLLISION_KINDS.get(str(props.get('nif_collision', 'MESH')), 'MESH'),
                                         verts, tri.reshape(-1, 3), props)
            result.append(d)
    return result
