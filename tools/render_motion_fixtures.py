"""Blender --background --factory-startup --python tools/render_motion_fixtures.py.

Writes known 3D, projected 2D, .blend files and silent PNG sequences to .cache/v03-truth.
The mannequin is a reproducible rendering fixture, not a photorealism benchmark.
"""
import json
from pathlib import Path
import sys
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.fixtures.motions import clip
from core import skeleton


def material(name, color):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1)
    return value


def main():
    output = ROOT / '.cache/v03-truth'
    output.mkdir(exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'MATERIAL'
    scene.display.shading.show_shadows = True
    scene.display.shading.background_type = 'WORLD'
    scene.world.color = (.15, .15, .15)
    scene.render.resolution_x, scene.render.resolution_y = 384, 512
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    mats = {'L': material('Shirt', (.16, .36, .55)), 'R': material('Shirt right', (.16, .36, .55)),
            'C': material('Body', (.16, .36, .55)), 'skin': material('Skin', (.68, .43, .28)),
            'dark': material('Trousers', (.07, .085, .11)), 'black': material('Hair eyes shoes', (.018, .012, .009))}
    base = clip('idle', 24)[0]['body3d']
    nodes, links = {}, []
    for name, point in base.items():
        radius = .105 if name == 'head' else .12 if name in ('pelvis', 'chest') else .09 if name == 'spine' else .05
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=radius, location=point)
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(mats['skin'] if name == 'head' or name.startswith(('wrist.', 'neck')) else mats['dark'] if name.startswith(('hip.', 'knee.', 'ankle.')) else mats['black'] if name.startswith(('toe.', 'heel.')) else mats['C'])
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        if name == 'head':
            obj.scale = (1, .9, 1.3)
        if name in ('spine', 'chest', 'pelvis'):
            obj.scale = (1.4, .9, 1.3)
        nodes[name] = obj
    for name, location, scale, mat in [('Eye L', (.037, -.092, .014), (.014, .01, .012), 'black'),
                                      ('Eye R', (-.037, -.092, .014), (.014, .01, .012), 'black'),
                                      ('Nose', (0, -.102, -.01), (.022, .035, .027), 'skin'),
                                      ('Hair', (0, .005, .063), (.103, .09, .074), 'black')]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=1)
        obj = bpy.context.object
        obj.name = name
        obj.parent = nodes['head']
        obj.location, obj.scale = location, scale
        obj.data.materials.append(mats[mat])
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    edges = [('pelvis', 'spine'), ('spine', 'chest'), ('chest', 'neck'), ('neck', 'head')]
    for side in ('L', 'R'):
        edges += [('pelvis', 'hip.' + side), ('chest', 'shoulder.' + side)]
        edges += [(a + '.' + side, b + '.' + side) for a, b in [('hip', 'knee'), ('knee', 'ankle'), ('ankle', 'toe'), ('ankle', 'heel'), ('shoulder', 'elbow'), ('elbow', 'wrist')]]
    for a, b in edges:
        radius = .115 if a in ('pelvis', 'spine') and b in ('spine', 'chest') else .07 if a.startswith(('hip.', 'knee.')) else .052
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=1)
        obj = bpy.context.object
        obj.name = a + ' to ' + b
        obj.data.materials.append(mats['dark'] if a.startswith(('hip.', 'knee.')) else mats['black'] if a.startswith('ankle.') else mats['skin'] if a.startswith('elbow.') else mats['C'])
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        links.append((a, b, obj))
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2.5
    for kind in ('walk', 'run', 'attack', 'idle'):
        frames = clip(kind, 24)
        folder = output / kind
        folder.mkdir(exist_ok=True)
        (folder / 'truth.json').write_text(json.dumps(dict(fps=24, coordinate_space='world', frames=frames)), encoding='utf-8')
        for obj in list(nodes.values()) + [link[2] for link in links]:
            obj.animation_data_clear()
        for frame in frames:
            points = frame['body3d']
            for name, obj in nodes.items():
                obj.location = points[name]
                obj.keyframe_insert('location', frame=frame['frame'])
            for a, b, obj in links:
                start, end = Vector(points[a]), Vector(points[b])
                obj.location = (start + end) / 2
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = (end - start).to_track_quat('Z', 'Y')
                obj.scale.z = (end - start).length
                for field in ('location', 'rotation_quaternion', 'scale'):
                    obj.keyframe_insert(field, frame=frame['frame'])
        scene.frame_start, scene.frame_end = 1, len(frames)
        for view, position in [('front', (0, -5, 1.1)), ('oblique', (3.6, -3.6, 1.1)), ('side', (5, 0, 1.1))]:
            camera.location = position
            camera.rotation_euler = (Vector((0, 0, .8)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()
            destination = folder / view
            destination.mkdir(exist_ok=True)
            projections = []
            for frame in frames:
                projection = {name: list(world_to_camera_view(scene, camera, Vector(point)))[:2] for name, point in frame['body3d'].items()}
                projections.append(dict(frame=frame['frame'], time=frame['time'], points=projection))
            (destination / 'projection.json').write_text(json.dumps(projections), encoding='utf-8')
            scene.render.filepath = str(destination / 'frame_')
            bpy.ops.render.render(animation=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / (kind + '.blend')))
        print('TRUTH READY', kind, flush=True)


if __name__ == '__main__':
    main()
