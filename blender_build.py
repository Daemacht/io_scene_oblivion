# SPDX-License-Identifier: GPL-3.0-or-later
"""Creates Blender data blocks from a :class:`nif_scene.SceneDesc` (Blender 4.0 API)."""
import os

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector


def _mat(m):
    return Matrix([list(map(float, row)) for row in m])


def _get_collection(context, name, cache):
    if name in cache:
        return cache[name]
    if name == 'main':
        col = context.collection
    else:
        col = bpy.data.collections.new(name.capitalize())
        context.scene.collection.children.link(col)
    cache[name] = col
    return col


# ----------------------------------------------------------------- materials
def _load_image(ts, colorspace):
    if ts.resolved:
        try:
            img = bpy.data.images.load(ts.resolved, check_existing=True)
        except RuntimeError:
            img = None
    else:
        img = None
    if img is None:
        name = os.path.basename(ts.path.replace('\\', '/'))
        img = bpy.data.images.get(name) or bpy.data.images.new(name, 4, 4)
        img.filepath = ts.path.replace('\\', '/')
        img.source = 'FILE'
    try:
        img.colorspace_settings.name = colorspace
    except TypeError:
        pass
    return img


def build_material(md):
    mat = bpy.data.materials.new(md.name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (600, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    links = nt.links

    # surface constants
    rough = (2.0 / (max(md.glossiness, 0.0) + 2.0)) ** 0.5
    bsdf.inputs['Roughness'].default_value = min(max(rough, 0.05), 1.0)
    spec = sum(md.specular) / 3.0
    bsdf.inputs['Specular IOR Level'].default_value = min(spec * 0.5, 1.0) if md.specular_enabled else 0.0
    bsdf.inputs['Base Color'].default_value = (*md.diffuse, 1.0)
    if any(c > 0 for c in md.emissive):
        bsdf.inputs['Emission Color'].default_value = (*md.emissive, 1.0)
        bsdf.inputs['Emission Strength'].default_value = 1.0
    bsdf.inputs['Alpha'].default_value = md.alpha

    def tex_node(slot, colorspace, y):
        ts = md.textures[slot]
        n = nt.nodes.new('ShaderNodeTexImage')
        n.image = _load_image(ts, colorspace)
        n.location = (-400, y)
        n.label = slot
        n.extension = 'EXTEND' if ts.clamp_mode == 0 else 'REPEAT'
        return n

    uv_nodes = {}

    def uv_for(slot, tex):
        ts = md.textures[slot]
        key = (ts.uv_set, tuple(sorted((ts.transform or {}).items())) if ts.transform else None)
        if key in uv_nodes:
            src = uv_nodes[key]
        else:
            uv = nt.nodes.new('ShaderNodeUVMap')
            uv.uv_map = 'UVMap' if ts.uv_set == 0 else 'UVMap.%03d' % ts.uv_set
            uv.location = (-900, 0)
            src = uv.outputs['UV']
            if ts.transform:
                mp = nt.nodes.new('ShaderNodeMapping')
                mp.location = (-700, 0)
                t = ts.transform
                mp.inputs['Location'].default_value = (t['translation'][0], -t['translation'][1], 0)
                mp.inputs['Scale'].default_value = (t['scale'][0], t['scale'][1], 1)
                mp.inputs['Rotation'].default_value = (0, 0, t['rotation'])
                links.new(src, mp.inputs['Vector'])
                src = mp.outputs['Vector']
            uv_nodes[key] = src
        links.new(src, tex.inputs['Vector'])

    color_out = None
    alpha_out = None
    y = 300
    if 'base' in md.textures:
        base = tex_node('base', 'sRGB', y)
        uv_for('base', base)
        color_out = base.outputs['Color']
        alpha_out = base.outputs['Alpha']
        if tuple(md.diffuse) != (1.0, 1.0, 1.0):
            mix = nt.nodes.new('ShaderNodeMix')
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.inputs[0].default_value = 1.0
            mix.inputs[7].default_value = (*md.diffuse, 1.0)
            mix.location = (-100, y)
            links.new(color_out, mix.inputs[6])
            color_out = mix.outputs[2]
    y -= 300
    if 'dark' in md.textures and color_out is not None:
        dk = tex_node('dark', 'sRGB', y)
        uv_for('dark', dk)
        mix = nt.nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = 1.0
        mix.location = (-100, y)
        links.new(color_out, mix.inputs[6])
        links.new(dk.outputs['Color'], mix.inputs[7])
        color_out = mix.outputs[2]
        y -= 300
    for slot in ('decal0', 'decal1', 'decal2', 'decal3'):
        if slot in md.textures and color_out is not None:
            dc = tex_node(slot, 'sRGB', y)
            uv_for(slot, dc)
            mix = nt.nodes.new('ShaderNodeMix')
            mix.data_type = 'RGBA'
            mix.blend_type = 'MIX'
            mix.location = (-100, y)
            links.new(dc.outputs['Alpha'], mix.inputs[0])
            links.new(color_out, mix.inputs[6])
            links.new(dc.outputs['Color'], mix.inputs[7])
            color_out = mix.outputs[2]
            y -= 300
    if color_out is not None:
        links.new(color_out, bsdf.inputs['Base Color'])
    if 'glow' in md.textures:
        gl = tex_node('glow', 'sRGB', y)
        uv_for('glow', gl)
        links.new(gl.outputs['Color'], bsdf.inputs['Emission Color'])
        bsdf.inputs['Emission Strength'].default_value = 1.0
        y -= 300
    if 'gloss' in md.textures:
        gs = tex_node('gloss', 'Non-Color', y)
        uv_for('gloss', gs)
        links.new(gs.outputs['Color'], bsdf.inputs['Specular IOR Level'])
        y -= 300
    nrm_slot = 'normal' if 'normal' in md.textures else ('bump' if 'bump' in md.textures else None)
    if nrm_slot:
        nm = tex_node(nrm_slot, 'Non-Color', y)
        uv_for(nrm_slot, nm)
        nn = nt.nodes.new('ShaderNodeNormalMap')
        nn.location = (0, y)
        links.new(nm.outputs['Color'], nn.inputs['Color'])
        links.new(nn.outputs['Normal'], bsdf.inputs['Normal'])

    # transparency
    if md.alpha_blend or md.alpha_test:
        if alpha_out is not None:
            links.new(alpha_out, bsdf.inputs['Alpha'])
        if hasattr(mat, 'blend_method'):
            if md.alpha_blend:
                mat.blend_method = 'BLEND'
                mat.show_transparent_back = False
            else:
                mat.blend_method = 'CLIP'
                mat.alpha_threshold = md.alpha_threshold
        if hasattr(mat, 'surface_render_method'):          # Blender 4.2+ EEVEE Next
            mat.surface_render_method = 'BLENDED' if md.alpha_blend else 'DITHERED'
    mat.use_backface_culling = not md.two_sided
    mat['nif_glossiness'] = md.glossiness
    mat['nif_alpha_blend'] = md.alpha_blend
    mat['nif_alpha_test'] = md.alpha_test
    for k, v in md.props.items():
        mat[k] = v
    return mat


# ------------------------------------------------------------------- meshes
def build_mesh(md, settings, material_cache):
    me = bpy.data.meshes.new(md.name)
    verts = md.vertices.astype(np.float32)
    faces = md.faces
    if md.convex_hull:
        bm = bmesh.new()
        for v in verts:
            bm.verts.new(Vector(v))
        bm.verts.ensure_lookup_table()
        bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
        # drop interior vertices left unused by the hull
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        bm.to_mesh(me)
        bm.free()
        me.update()
        return me

    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    if len(me.polygons) != len(faces):          # should not happen (degenerates are filtered)
        return me

    me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), dtype=bool))
    loop_vi = faces.reshape(-1)
    for k, uv in enumerate(md.uv_sets):
        layer = me.uv_layers.new(name='UVMap' if k == 0 else 'UVMap.%03d' % k)
        layer.data.foreach_set('uv', uv[loop_vi].astype(np.float32).reshape(-1))
    if md.colors is not None:
        attr = me.color_attributes.new(name='Col', type='FLOAT_COLOR', domain='POINT')
        attr.data.foreach_set('color', md.colors.astype(np.float32).reshape(-1))
    if md.normals is not None and len(md.normals) == len(verts):
        n = md.normals.astype(np.float32)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        if hasattr(me, 'use_auto_smooth'):          # Blender < 4.1 needs this for custom normals
            me.use_auto_smooth = True
        me.normals_split_custom_set_from_vertices((n / ln).tolist())
    if md.material is not None:
        mat = material_cache.get(md.material.key)
        if mat is None:
            mat = build_material(md.material)
            material_cache[md.material.key] = mat
        me.materials.append(mat)
    return me


# ----------------------------------------------------------------- armature
def build_armature(od, context, col):
    data = bpy.data.armatures.new(od.name)
    data.display_type = 'STICK'
    obj = bpy.data.objects.new(od.name, data)
    col.objects.link(obj)
    obj.show_in_front = True
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit = {}
        for b in od.bones:
            eb = data.edit_bones.new(b.name)
            eb.head = (0.0, 0.0, 0.0)
            eb.tail = (0.0, 1.0, 0.0)
            eb.matrix = _mat(b.matrix)
            eb.length = b.length
            edit[b.name] = eb
        for b in od.bones:
            if b.parent and b.parent in edit and b.parent != b.name:
                edit[b.name].parent = edit[b.parent]
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)
    context.view_layer.objects.active = prev_active
    return obj


# ----------------------------------------------------------------- top level
def build_scene(desc, context, settings):
    """Instantiate ``desc`` in the current scene.  Returns list of created objects."""
    collections = {}
    objects = {}
    material_cache = {}
    created = []
    for od in desc.objects:
        col = _get_collection(context, od.collection, collections)
        if od.kind == 'ARMATURE':
            obj = build_armature(od, context, col)
        elif od.kind == 'MESH':
            me = build_mesh(od.mesh, settings, material_cache)
            obj = bpy.data.objects.new(od.name, me)
            col.objects.link(obj)
            if od.display:
                obj.display_type = od.display
            if od.mesh.skin is not None and od.armature is not None:
                for bone_name, (idx, w) in od.mesh.skin.weights.items():
                    vg = obj.vertex_groups.new(name=bone_name)
                    for i, wt in zip(idx.tolist(), w.tolist()):
                        vg.add([i], wt, 'REPLACE')
                mod = obj.modifiers.new('Armature', 'ARMATURE')
                mod.object = objects[id(od.armature)]
        else:
            obj = bpy.data.objects.new(od.name, None)
            obj.empty_display_type = 'PLAIN_AXES'
            obj.empty_display_size = 0.5
            col.objects.link(obj)
        objects[id(od)] = obj
        if od.parent is not None:
            obj.parent = objects[id(od.parent)]
        obj.matrix_basis = _mat(od.matrix)
        for k, v in od.props.items():
            obj[k] = v
        if od.hidden:
            obj.hide_set(True)
        created.append(obj)
    return created
