# SPDX-License-Identifier: GPL-3.0-or-later
# Oblivion NIF importer for Blender 4.0.  Block layouts come from niftools' nif.xml (GPL-3.0).

bl_info = {
    "name": "Oblivion NIF Importer/Exporter",
    "author": "Daemacht",
    "version": (0, 4, 0),
    "blender": (4, 0, 0),
    "location": "File > Import / Export > Oblivion NIF (.nif)",
    "description": "Imports and exports TES IV: Oblivion .nif meshes, materials, skeletons and skinning",
    "category": "Import-Export",
}


def register():
    from . import blender_ui
    blender_ui.register()


def unregister():
    from . import blender_ui
    blender_ui.unregister()
