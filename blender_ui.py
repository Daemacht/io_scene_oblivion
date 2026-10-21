# SPDX-License-Identifier: GPL-3.0-or-later
import os

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import blender_build, nif_reader, nif_scene


class IMPORT_OT_oblivion_nif(bpy.types.Operator, ImportHelper):
    """Import an Oblivion .nif file"""
    bl_idname = "import_scene.oblivion_nif"
    bl_label = "Import Oblivion NIF"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".nif"
    filter_glob: StringProperty(default="*.nif", options={'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN'})

    scale: FloatProperty(name="Scale", default=0.1, min=0.0001, max=100.0,
                         description="NIF units to Blender units (0.1 matches the NifTools add-on)")
    texture_dir: StringProperty(name="Texture Folder", subtype='DIR_PATH', default="",
                                description="Oblivion 'Data' folder (the one containing 'textures'). "
                                            "Optional: it is also guessed from the .nif location")
    import_skeleton: BoolProperty(name="Armature & Skinning", default=True,
                                  description="Create an armature and vertex groups for skinned meshes")
    import_normals: BoolProperty(name="Custom Normals", default=True)
    import_vertex_colors: BoolProperty(name="Vertex Colors", default=True)
    find_normal_maps: BoolProperty(name="Find _n Normal Maps", default=True,
                                   description="Use <texture>_n.dds next to the diffuse texture if it exists")
    import_collision: BoolProperty(name="Havok Collision", default=False,
                                   description="Import collision shapes as wire objects in a 'Collision' collection")
    import_empties: BoolProperty(name="Keep NiNode Hierarchy", default=True,
                                 description="Create empties for NiNodes")

    def execute(self, context):
        settings = nif_scene.ImportSettings(
            scale=self.scale, texture_dir=self.texture_dir, import_collision=self.import_collision,
            import_skeleton=self.import_skeleton, import_normals=self.import_normals,
            import_vertex_colors=self.import_vertex_colors, find_normal_maps=self.find_normal_maps,
            import_empties=self.import_empties)
        paths = [os.path.join(self.directory, f.name) for f in self.files] if self.files else [self.filepath]
        total = 0
        for path in paths:
            if not os.path.isfile(path):
                continue
            try:
                nif = nif_reader.read_nif(path)
                desc = nif_scene.build_scene(nif, settings, path)
                objs = blender_build.build_scene(desc, context, settings)
            except nif_reader.NifError as e:
                self.report({'ERROR'}, "%s: %s" % (os.path.basename(path), e))
                continue
            total += len(objs)
            for w in desc.warnings:
                self.report({'WARNING'}, "%s: %s" % (os.path.basename(path), w))
        self.report({'INFO'}, "Imported %d objects" % total)
        return {'FINISHED'} if total else {'CANCELLED'}


class EXPORT_OT_oblivion_nif(bpy.types.Operator, ExportHelper):
    """Export to an Oblivion .nif file (NIF 20.0.0.5)"""
    bl_idname = "export_scene.oblivion_nif"
    bl_label = "Export Oblivion NIF"
    bl_options = {'REGISTER', 'PRESET'}

    filename_ext = ".nif"
    filter_glob: StringProperty(default="*.nif", options={'HIDDEN'})

    scale: FloatProperty(name="Scale", default=0.1, min=0.0001, max=100.0,
                         description="Same value used on import: Blender units = NIF units * Scale")
    selected_only: BoolProperty(name="Selected Only", default=True)
    export_normals: BoolProperty(name="Normals", default=True)
    export_vertex_colors: BoolProperty(name="Vertex Colors", default=True)
    export_collision: BoolProperty(
        name="Havok Collision", default=True,
        description="Export meshes in the 'Collision' collection (or with a nif_collision property) as Havok "
                    "collision. Unedited imported meshes keep their original MOPP; new or edited meshes use "
                    "bhkNiTriStripsShape. Set the custom property nif_collision to BOX, SPHERE, CAPSULE, CONVEX or MESH")
    tangents: EnumProperty(
        name="Tangent Space", default='AUTO',
        items=[('AUTO', "Auto", "Write tangent space whenever the mesh has UVs and normals (what Oblivion meshes normally carry)"),
               ('ALWAYS', "Always", "Always write tangent space data"),
               ('NEVER', "Never", "Never write tangent space data")])

    prn: EnumProperty(
        name="Attach Point (Prn)", default='NONE',
        description="Adds a 'Prn' string extra data to the root so the item attaches to that body part "
                    "(ignored if the root already has a nif_extra_Prn property)",
        items=[('NONE', "None", ""), ('BACK', "Back weapon", "BackWeapon"), ('SIDE', "Side weapon", "SideWeapon"),
               ('QUIVER', "Quiver", "Quiver"), ('SHIELD', "Shield", "Bip01 L ForearmTwist"),
               ('HELM', "Helmet", "Bip01 Head"), ('RING', "Ring", "Bip01 R Finger1")])

    def execute(self, context):
        from . import blender_export, nif_export
        try:
            objs = blender_export.collect(context, dict(
                selected_only=self.selected_only, normals=self.export_normals, colors=self.export_vertex_colors,
                collision=self.export_collision))
            if not objs:
                self.report({'ERROR'}, "Nothing to export (select objects or disable 'Selected Only')")
                return {'CANCELLED'}
            data, warnings = nif_export.export_bytes(
                objs, nif_export.ExportSettings(scale=self.scale, tangents=self.tangents, prn=self.prn,
                                          collision=self.export_collision))
        except nif_export.NifExportError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        with open(self.filepath, 'wb') as fh:
            fh.write(data)
        for w in warnings:
            self.report({'WARNING'}, w)
        self.report({'INFO'}, "Exported %s (%d bytes)" % (os.path.basename(self.filepath), len(data)))
        return {'FINISHED'}


def menu_import(self, context):
    self.layout.operator(IMPORT_OT_oblivion_nif.bl_idname, text="Oblivion NIF (.nif)")


def menu_export(self, context):
    self.layout.operator(EXPORT_OT_oblivion_nif.bl_idname, text="Oblivion NIF (.nif)")


_classes = (IMPORT_OT_oblivion_nif, EXPORT_OT_oblivion_nif)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
