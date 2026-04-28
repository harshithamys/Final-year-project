# Blender Python Script for UHI Hotspot Visualization
# Run this script in Blender (Text Editor > Run Script)
# Requires Blender 2.8+

import bpy
import bmesh
import math
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Hotspot data
hotspot_data = [{'position': {'x': 0.0, 'y': 0.0, 'z': 30.64438841363893}, 'uhi_value': 0.03171157540733027, 'intensity': 0.6128877682727786, 'color': {'r': np.float64(0.4515510730911143), 'g': 1, 'b': 0}}, {'position': {'x': 0.0, 'y': 1.1111111111112848, 'z': 48.46560043671285}, 'uhi_value': 0.11148881098461108, 'intensity': 0.9693120087342569, 'color': {'r': 1, 'g': np.float64(0.12275196506297226), 'b': 0}}, {'position': {'x': 0.0, 'y': 2.2222222222225696, 'z': 45.71836150258204}, 'uhi_value': 0.09919070649253356, 'intensity': 0.9143672300516408, 'color': {'r': 1, 'g': np.float64(0.3425310797934369), 'b': 0}}, {'position': {'x': 0.0, 'y': 3.333333333333144, 'z': 25.101924471569344}, 'uhi_value': 0.006900555202162288, 'intensity': 0.5020384894313868, 'color': {'r': np.float64(0.00815395772554739), 'g': 1, 'b': 0}}, {'position': {'x': 0.0, 'y': 4.444444444444429, 'z': 8.296890876718013}, 'uhi_value': -0.06832772457166911, 'intensity': 0.16593781753436027, 'color': {'r': 0, 'g': np.float64(0.6637512701374411), 'b': 1}}, {'position': {'x': 0.0, 'y': 5.5555555555557135, 'z': 38.006279805449864}, 'uhi_value': 0.0646673218009223, 'intensity': 0.7601255961089973, 'color': {'r': 1, 'g': np.float64(0.9594976155640107), 'b': 0}}, {'position': {'x': 0.0, 'y': 6.666666666666998, 'z': 17.954442782353013}, 'uhi_value': -0.02509537918022655, 'intensity': 0.3590888556470602, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.5636445774117591)}}, {'position': {'x': 0.0, 'y': 7.7777777777775725, 'z': 35.469825331934125}, 'uhi_value': 0.053312800806350195, 'intensity': 0.7093965066386825, 'color': {'r': np.float64(0.8375860265547299), 'g': 1, 'b': 0}}, {'position': {'x': 0.0, 'y': 8.888888888888857, 'z': 50.0}, 'uhi_value': 0.11835760056365577, 'intensity': 1.0, 'color': {'r': 1, 'g': 0.0, 'b': 0}}, {'position': {'x': 0.0, 'y': 10.000000000000142, 'z': 41.112265288275665}, 'uhi_value': 0.07857136689292765, 'intensity': 0.8222453057655132, 'color': {'r': 1, 'g': np.float64(0.711018776937947), 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 0.0, 'z': 12.654462485071921}, 'uhi_value': -0.04882091345834968, 'intensity': 0.2530892497014384, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.9876430011942463)}}, {'position': {'x': 1.1111111111105743, 'y': 1.1111111111112848, 'z': 46.02946879375956}, 'uhi_value': 0.10058338840585934, 'intensity': 0.9205893758751913, 'color': {'r': 1, 'g': np.float64(0.3176424964992348), 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 2.2222222222225696, 'z': 18.886095747687435}, 'uhi_value': -0.020924804362109528, 'intensity': 0.3777219149537487, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.48911234018500527)}}, {'position': {'x': 1.1111111111105743, 'y': 3.333333333333144, 'z': 27.86737773060325}, 'uhi_value': 0.0192801967124176, 'intensity': 0.557347554612065, 'color': {'r': np.float64(0.2293902184482599), 'g': 1, 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 4.444444444444429, 'z': 12.081373219616756}, 'uhi_value': -0.05138636620521294, 'intensity': 0.24162746439233512, 'color': {'r': 0, 'g': np.float64(0.9665098575693405), 'b': 1}}, {'position': {'x': 1.1111111111105743, 'y': 5.5555555555557135, 'z': 14.456600368561185}, 'uhi_value': -0.04075358457590778, 'intensity': 0.2891320073712237, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.8434719705151053)}}, {'position': {'x': 1.1111111111105743, 'y': 6.666666666666998, 'z': 17.414547634088336}, 'uhi_value': -0.027512237381249884, 'intensity': 0.34829095268176674, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.606836189272933)}}, {'position': {'x': 1.1111111111105743, 'y': 7.7777777777775725, 'z': 29.693442277289684}, 'uhi_value': 0.027454634081307442, 'intensity': 0.5938688455457937, 'color': {'r': np.float64(0.37547538218317467), 'g': 1, 'b': 0}}, {'position': {'x': 1.1111111111105743, 'y': 8.888888888888857, 'z': 10.310305018205641}, 'uhi_value': -0.05931461064473085, 'intensity': 0.20620610036411283, 'color': {'r': 0, 'g': np.float64(0.8248244014564513), 'b': 1}}, {'position': {'x': 1.1111111111105743, 'y': 10.000000000000142, 'z': 18.73502569084747}, 'uhi_value': -0.02160107438782894, 'intensity': 0.37470051381694935, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.5011979447322026)}}, {'position': {'x': 2.2222222222225696, 'y': 0.0, 'z': 17.979520697621147}, 'uhi_value': -0.02498311707641774, 'intensity': 0.35959041395242297, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.5616383441903081)}}, {'position': {'x': 2.2222222222225696, 'y': 1.1111111111112848, 'z': 11.798210052140501}, 'uhi_value': -0.052653955339403924, 'intensity': 0.23596420104281002, 'color': {'r': 0, 'g': np.float64(0.9438568041712401), 'b': 1}}, {'position': {'x': 2.2222222222225696, 'y': 2.2222222222225696, 'z': 33.62966373133776}, 'uhi_value': 0.045075257516871914, 'intensity': 0.6725932746267551, 'color': {'r': np.float64(0.6903730985070204), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 3.333333333333144, 'z': 34.774702646556285}, 'uhi_value': 0.05020106148386756, 'intensity': 0.6954940529311257, 'color': {'r': np.float64(0.7819762117245026), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 4.444444444444429, 'z': 43.69549687071277}, 'uhi_value': 0.09013528713785447, 'intensity': 0.8739099374142555, 'color': {'r': 1, 'g': np.float64(0.504360250342978), 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 5.5555555555557135, 'z': 36.55154791000796}, 'uhi_value': 0.05815516712961308, 'intensity': 0.7310309582001593, 'color': {'r': np.float64(0.9241238328006371), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 6.666666666666998, 'z': 2.877951013363275}, 'uhi_value': -0.092585785218624, 'intensity': 0.0575590202672655, 'color': {'r': 0, 'g': np.float64(0.230236081069062), 'b': 1}}, {'position': {'x': 2.2222222222225696, 'y': 7.7777777777775725, 'z': 28.334600321964476}, 'uhi_value': 0.02137173384747846, 'intensity': 0.5666920064392895, 'color': {'r': np.float64(0.2667680257571581), 'g': 1, 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 8.888888888888857, 'z': 37.855534392782545}, 'uhi_value': 0.06399250505540663, 'intensity': 0.7571106878556509, 'color': {'r': 1, 'g': np.float64(0.9715572485773962), 'b': 0}}, {'position': {'x': 2.2222222222225696, 'y': 10.000000000000142, 'z': 9.669799073967797}, 'uhi_value': -0.062181856348106836, 'intensity': 0.19339598147935594, 'color': {'r': 0, 'g': np.float64(0.7735839259174238), 'b': 1}}, {'position': {'x': 3.333333333333144, 'y': 0.0, 'z': 34.75352831920711}, 'uhi_value': 0.05010627391841249, 'intensity': 0.6950705663841422, 'color': {'r': np.float64(0.7802822655365689), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 1.1111111111112848, 'z': 29.87812693448423}, 'uhi_value': 0.02828138095943458, 'intensity': 0.5975625386896846, 'color': {'r': np.float64(0.39025015475873825), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 2.2222222222225696, 'z': 25.38970717394504}, 'uhi_value': 0.008188823834664235, 'intensity': 0.5077941434789008, 'color': {'r': np.float64(0.031176573915603356), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 3.333333333333144, 'z': 38.27319351389104}, 'uhi_value': 0.06586216970293632, 'intensity': 0.7654638702778208, 'color': {'r': 1, 'g': np.float64(0.9381445188887167), 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 4.444444444444429, 'z': 36.25542185988264}, 'uhi_value': 0.05682954922939222, 'intensity': 0.7251084371976527, 'color': {'r': np.float64(0.9004337487906109), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 5.5555555555557135, 'z': 39.687719235398475}, 'uhi_value': 0.07219434012819777, 'intensity': 0.7937543847079696, 'color': {'r': 1, 'g': np.float64(0.8249824611681218), 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 6.666666666666998, 'z': 28.087230214221513}, 'uhi_value': 0.020264373509997685, 'intensity': 0.5617446042844303, 'color': {'r': np.float64(0.24697841713772117), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 7.7777777777775725, 'z': 25.45336040162138}, 'uhi_value': 0.008473769579825405, 'intensity': 0.5090672080324277, 'color': {'r': np.float64(0.03626883212971066), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 8.888888888888857, 'z': 36.52838211091618}, 'uhi_value': 0.05805146467609375, 'intensity': 0.7305676422183236, 'color': {'r': np.float64(0.9222705688732944), 'g': 1, 'b': 0}}, {'position': {'x': 3.333333333333144, 'y': 10.000000000000142, 'z': 20.95398343834955}, 'uhi_value': -0.011667837819558088, 'intensity': 0.419079668766991, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.32368132493203605)}}, {'position': {'x': 4.444444444445139, 'y': 0.0, 'z': 0.0}, 'uhi_value': -0.10546902658715096, 'intensity': 0.0, 'color': {'r': 0, 'g': 0, 'b': 1}}, {'position': {'x': 4.444444444445139, 'y': 1.1111111111112848, 'z': 31.665625789167333}, 'uhi_value': 0.036283177753027585, 'intensity': 0.6333125157833467, 'color': {'r': np.float64(0.5332500631333867), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 2.2222222222225696, 'z': 14.07666665707418}, 'uhi_value': -0.042454370199568256, 'intensity': 0.2815333331414836, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.8738666674340656)}}, {'position': {'x': 4.444444444445139, 'y': 3.333333333333144, 'z': 34.78596157837118}, 'uhi_value': 0.050251462458536525, 'intensity': 0.6957192315674235, 'color': {'r': np.float64(0.7828769262696942), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 4.444444444444429, 'z': 25.627289165838334}, 'uhi_value': 0.00925236735300915, 'intensity': 0.5125457833167667, 'color': {'r': np.float64(0.050183133267066804), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 5.5555555555557135, 'z': 30.063156348496562}, 'uhi_value': 0.02910967115267602, 'intensity': 0.6012631269699312, 'color': {'r': np.float64(0.4050525078797249), 'g': 1, 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 6.666666666666998, 'z': 22.191398008206832}, 'uhi_value': -0.006128511228389643, 'intensity': 0.44382796016413667, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.22468815934345332)}}, {'position': {'x': 4.444444444445139, 'y': 7.7777777777775725, 'z': 21.875155445538855}, 'uhi_value': -0.007544181351659676, 'intensity': 0.4375031089107771, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.24998756435689162)}}, {'position': {'x': 4.444444444445139, 'y': 8.888888888888857, 'z': 39.794327838706856}, 'uhi_value': 0.07267157701027319, 'intensity': 0.7958865567741371, 'color': {'r': 1, 'g': np.float64(0.8164537729034516), 'b': 0}}, {'position': {'x': 4.444444444445139, 'y': 10.000000000000142, 'z': 23.76919253223058}, 'uhi_value': 0.0009345373045953126, 'intensity': 0.4753838506446116, 'color': {'r': 0, 'g': 1, 'b': np.float64(0.09846459742155367)}}]

# Scene settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128

# Create terrain plane
bpy.ops.mesh.primitive_plane_add(size=200, location=(50, 50, 0))
terrain = bpy.context.active_object
terrain.name = "UHI_Terrain"

# Create terrain material
terrain_mat = bpy.data.materials.new(name="TerrainMaterial")
terrain_mat.use_nodes = True
terrain_bsdf = terrain_mat.node_tree.nodes["Principled BSDF"]
terrain_bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.18, 1)
terrain_bsdf.inputs["Roughness"].default_value = 0.8
terrain.data.materials.append(terrain_mat)

# Subdivide terrain for detail
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=20)
bpy.ops.object.mode_set(mode='OBJECT')

# Create hotspot pillar material
def create_hotspot_material(name, color, emission_strength):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    emission = nodes.new('ShaderNodeEmission')
    mix = nodes.new('ShaderNodeMixShader')
    
    # Set node positions
    output.location = (400, 0)
    mix.location = (200, 0)
    principled.location = (0, 100)
    emission.location = (0, -100)
    
    # Configure nodes
    principled.inputs["Base Color"].default_value = (*color, 1)
    principled.inputs["Roughness"].default_value = 0.4
    principled.inputs["Metallic"].default_value = 0.6
    
    emission.inputs["Color"].default_value = (*color, 1)
    emission.inputs["Strength"].default_value = emission_strength
    
    mix.inputs["Fac"].default_value = 0.3
    
    # Link nodes
    links.new(principled.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    
    return mat

# Create hotspot pillars
pillar_collection = bpy.data.collections.new("UHI_Hotspots")
bpy.context.scene.collection.children.link(pillar_collection)

for i, point in enumerate(hotspot_data):
    pos = point['position']
    color = (point['color']['r'], point['color']['g'], point['color']['b'])
    height = max(point['position']['z'], 1)
    intensity = point['intensity']
    
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=1.5,
        depth=height,
        location=(pos['x'], pos['y'], height/2)
    )
    pillar = bpy.context.active_object
    pillar.name = f"Hotspot_{i:04d}"
    
    # Apply material
    mat_name = f"HotspotMat_{i:04d}"
    emission = 2.0 if intensity > 0.7 else 0.5
    mat = create_hotspot_material(mat_name, color, emission)
    pillar.data.materials.append(mat)
    
    # Move to collection
    bpy.context.scene.collection.objects.unlink(pillar)
    pillar_collection.objects.link(pillar)
    
    # Add glow ring for high intensity
    if intensity > 0.7:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=2.5,
            minor_radius=0.2,
            location=(pos['x'], pos['y'], 0.2)
        )
        glow = bpy.context.active_object
        glow.name = f"Glow_{i:04d}"
        
        glow_mat = bpy.data.materials.new(name=f"GlowMat_{i:04d}")
        glow_mat.use_nodes = True
        glow_bsdf = glow_mat.node_tree.nodes["Principled BSDF"]
        glow_bsdf.inputs["Emission"].default_value = (*color, 1)
        glow_bsdf.inputs["Emission Strength"].default_value = 5
        glow.data.materials.append(glow_mat)
        
        bpy.context.scene.collection.objects.unlink(glow)
        pillar_collection.objects.link(glow)

# Add lighting
# Sun light
bpy.ops.object.light_add(type='SUN', location=(100, 100, 100))
sun = bpy.context.active_object
sun.name = "UHI_Sun"
sun.data.energy = 3

# Area light for fill
bpy.ops.object.light_add(type='AREA', location=(0, 0, 80))
area = bpy.context.active_object
area.name = "UHI_AreaLight"
area.data.energy = 500
area.data.size = 50

# Add camera
bpy.ops.object.camera_add(location=(150, -100, 100))
camera = bpy.context.active_object
camera.name = "UHI_Camera"
camera.rotation_euler = (math.radians(60), 0, math.radians(45))
bpy.context.scene.camera = camera

# Add HDRI world (optional - uses solid color if no HDRI)
world = bpy.data.worlds.new(name="UHI_World")
bpy.context.scene.world = world
world.use_nodes = True
world_nodes = world.node_tree.nodes
bg_node = world_nodes["Background"]
bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.1, 1)
bg_node.inputs["Strength"].default_value = 0.5

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = False

print(f"Created {len(hotspot_data)} hotspot pillars")
print("Scene setup complete! Press F12 to render.")
