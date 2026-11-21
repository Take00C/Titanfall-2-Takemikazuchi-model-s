"""Stage 30 (runs inside Blender) - review renders of a cleaned model.

    blender -b --factory-startup --python render_preview.py -- <clean_dir> <out_dir> [--lod 0] [--samples 64]

Builds the scene straight from build/clean/<unit>/ (lod<N>.npz + manifest.json +
textures/) - no intermediate .blend or FBX - and writes a turnaround:
front, back, side, two three-quarter views, an upper-body close-up, and one
untextured "clay" view that shows the geometry without the paint.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

def args():
    """<clean_dir> <out_dir> [--lod N] [--samples N] [--res N]
       [--glow r,g,b]   linear RGB; overrides the colour extracted from the game
       [--only a,b]     render only these views        [--suffix _x]  appended to file names"""
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = {"lod": 0, "samples": 64, "res": 1500, "glow": None, "only": None, "suffix": "",
           "device": "cpu", "threads": 4}
    pos = []
    i = 0
    while i < len(a):
        if a[i].startswith("--"):
            k, v = a[i][2:], a[i + 1]
            if k == "glow":
                opt[k] = tuple(float(x) for x in v.split(","))
            elif k == "only":
                opt[k] = set(v.split(","))
            elif k in ("suffix", "device"):
                opt[k] = v
            else:
                opt[k] = int(v)
            i += 2
        else:
            pos.append(a[i]); i += 1
    return pos[0], pos[1], opt


def socket(node, ident, output=False):
    for s in (node.outputs if output else node.inputs):
        if s.identifier == ident:
            return s
    raise KeyError(f"{node.bl_idname} has no socket {ident}")


def new_material(name):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf


def textured_material(name, maps, clean_dir):
    mat, nt, bsdf = new_material(name)

    def img(key, non_color):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.abspath(os.path.join(clean_dir, maps[key]["file"])))
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        return n

    if "bc" in maps:
        col = img("bc", False).outputs["Color"]
        if "ao" in maps:
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
            socket(mix, "Factor_Float").default_value = 1.0
            nt.links.new(col, socket(mix, "A_Color"))
            nt.links.new(img("ao", True).outputs["Color"], socket(mix, "B_Color"))
            col = socket(mix, "Result_Color", output=True)
        nt.links.new(col, bsdf.inputs["Base Color"])
    if "mt" in maps:
        nt.links.new(img("mt", True).outputs["Color"], bsdf.inputs["Metallic"])
    if "rh" in maps:
        nt.links.new(img("rh", True).outputs["Color"], bsdf.inputs["Roughness"])
    if "nm" in maps:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(img("nm", True).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def glow_material(emissive_linear):
    """The unit's lit panels.  In the game this is a colour-only shader: the panel shows its
    emissive colour whatever the lighting, so strength 1.0 puts exactly that colour on screen."""
    mat, nt, bsdf = new_material("glow")
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Emission Color"].default_value = (*emissive_linear, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def flat_material(name, rgb, rough=0.7, metal=0.0):
    mat, nt, bsdf = new_material(name)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return mat


def build_model(clean_dir, lod, glow_override=None):
    man = json.load(open(os.path.join(clean_dir, "manifest.json")))
    L = man["lods"][str(lod)]
    z = np.load(os.path.join(clean_dir, L["file"]))
    mats = {s: textured_material(s, maps, clean_dir) for s, maps in man["texture_sets"].items()}
    if glow_override is not None:
        glow_rgb = glow_override
        print("glow colour: override", glow_rgb)
    elif man.get("glow"):
        glow_rgb = tuple(man["glow"]["emissive_linear"])
        print("glow colour: extracted from the game,", glow_rgb)
    else:
        glow_rgb = (1.0, 0.0, 1.0)            # no glow material in the manifest: make that obvious
        print("glow colour: MISSING from manifest, rendering magenta")
    glow = glow_material(glow_rgb)
    objs = []
    for part, info in L["parts"].items():
        pos, nrm, uv = z[f"{part}.pos"], z[f"{part}.nrm"], z[f"{part}.uv"]
        tri, mi = z[f"{part}.tri"], z[f"{part}.mat"]
        me = bpy.data.meshes.new(part)
        me.from_pydata(pos.tolist(), [], tri.tolist())
        me.update()
        if len(me.polygons) != len(tri):
            raise RuntimeError(f"{part}: Blender kept {len(me.polygons)} of {len(tri)} triangles")
        lv = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", lv)
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", uv[lv].astype(np.float64).ravel())
        me.materials.append(mats.get(info["texset"]) or flat_material("missing_" + part, (1, 0, 1)))
        me.materials.append(glow)
        me.polygons.foreach_set("material_index", mi.astype(np.int32))
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
        me.normals_split_custom_set_from_vertices(nrm.astype(np.float64).tolist())
        ob = bpy.data.objects.new(part, me)
        bpy.context.collection.objects.link(ob)
        objs.append(ob)
    return man, objs


def setup_render(opt):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = opt["samples"]
    scene.cycles.use_denoising = True
    scene.render.resolution_x = int(opt["res"] * 0.8)
    scene.render.resolution_y = opt["res"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"       # keep the paint colour honest
    scene.view_settings.exposure = 0.0
    # CPU on a limited number of threads by default; the GPU only when asked for (--device gpu)
    device = f"CPU, {opt['threads']} threads"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = opt["threads"]
    try:
        if opt["device"] != "gpu":
            raise RuntimeError("cpu requested")
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type != "CPU"]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type != "CPU"
                    scene.cycles.device = "GPU"
                    device = f"{kind}: {gpus[0].name}"
                    break
            except Exception:
                continue
    except Exception:
        pass
    print("render device:", device)
    return scene


def setup_stage(scene, height):
    world = bpy.data.worlds.new("world")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (0.62, 0.66, 0.72, 1)
    bg.inputs[1].default_value = 0.55
    out = world.node_tree.nodes.get("World Output") or world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs[0], out.inputs[0])
    # key / fill / rim, turned with the camera so every view is lit the same way
    rig = bpy.data.objects.new("light_rig", None)
    bpy.context.collection.objects.link(rig)
    for name, az, el, energy in (("key", 35, 50, 2.6), ("fill", -70, 25, 0.9), ("rim", 160, 40, 1.6)):
        sun = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        sun.data.energy = energy
        sun.data.angle = math.radians(6)
        sun.rotation_euler = (math.radians(90 - el), 0, math.radians(az + 90))
        sun.parent = rig
        bpy.context.collection.objects.link(sun)
    floor_me = bpy.data.meshes.new("floor")
    s = height * 6
    floor_me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("floor", floor_me)
    floor_me.materials.append(flat_material("floor", (0.16, 0.17, 0.19), rough=0.9))
    bpy.context.collection.objects.link(floor)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = 85
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36
    cam.data.clip_end = 2000
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    return cam, rig


def main():
    clean_dir, out_dir, opt = args()
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = setup_render(opt)
    man, objs = build_model(clean_dir, opt["lod"], opt["glow"])
    b = man["bounds_m"]
    H, W, Lx = b["height_z"], b["width_y"], b["length_x"]
    cam, rig = setup_stage(scene, H)
    half_v = math.atan(18 / 85)                                   # half the vertical field of view

    def shoot(tag, az_deg, target_z, frame_h, elev_deg=6.0):
        """camera on a circle around the model: az 0 = from the front (+X side), 90 = from its left"""
        if opt["only"] and tag not in opt["only"]:
            return
        tag += opt["suffix"]
        d = (frame_h / 2) / math.tan(half_v)
        az, el = math.radians(az_deg), math.radians(elev_deg)
        target = Vector((0, 0, target_z))
        cam.location = target + Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))) * d
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        rig.rotation_euler = (0, 0, az)
        scene.render.filepath = os.path.join(out_dir, tag + ".png")
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)

    full = max(H * 1.12, max(W, Lx) * 1.12 / 0.8)
    shoot("front", 0, H * 0.5, full)
    shoot("threequarter_front", 35, H * 0.5, full)
    shoot("side_left", 90, H * 0.5, full)
    shoot("threequarter_back", 145, H * 0.5, full)
    shoot("back", 180, H * 0.5, full)
    shoot("upper_body", 25, H * 0.80, H * 0.50, elev_deg=2.0)

    if opt["only"] and "clay_threequarter_front" not in opt["only"]:
        return
    clay = flat_material("clay", (0.60, 0.60, 0.62), rough=0.55)
    for ob in objs:
        for i in range(len(ob.data.materials)):
            ob.data.materials[i] = clay
    shoot("clay_threequarter_front", 35, H * 0.5, full)


if __name__ == "__main__":
    main()
