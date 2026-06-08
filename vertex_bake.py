import bpy
import os
import mathutils
import array

# --- USER CONFIGURATION ---
TERRAIN_NAME = "Terrain_Mesh"
EXPORT_PATH = r"C:\Users\evane\OneDrive\Desktop\CREATE\Map_Art\data\wurl\terrain_shading_base.ply"
# --------------------------

terrain_obj = bpy.data.objects.get(TERRAIN_NAME)
if not terrain_obj:
    raise ValueError(f"Could not find object named '{TERRAIN_NAME}'.")

mesh = terrain_obj.data
num_loops = len(mesh.loops)
print(f"Initializing process for {num_loops:,} loop corners...")

# Ensure export directory exists
export_dir = os.path.dirname(EXPORT_PATH)
if export_dir and not os.path.exists(export_dir):
    os.makedirs(export_dir)

# 1. Force Engine Settings to Cycles GPU
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.use_adaptive_sampling = False  
scene.cycles.samples = 50                   

# 2. Force Shade Smooth
mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
mesh.update()

# 3. Create or completely clean target attribute layer
attr_name = "shading_pack"
if attr_name not in mesh.attributes:
    color_attr = mesh.attributes.new(name=attr_name, type='BYTE_COLOR', domain='CORNER')
else:
    color_attr = mesh.attributes[attr_name]
mesh.attributes.active = color_attr

# 4. Establish Selection Context
bpy.ops.object.select_all(action='DESELECT')
terrain_obj.select_set(True)
bpy.context.view_layer.objects.active = terrain_obj


# --- BAKE PASS 1: PURE DIFFUSE SHADOWS ---
print("Baking Pass 1: Pure Diffuse Shadows...")
scene.cycles.bake_type = 'DIFFUSE'
scene.render.bake.target = 'VERTEX_COLORS'
scene.render.bake.use_pass_direct = True    
scene.render.bake.use_pass_indirect = True  
scene.render.bake.use_pass_color = False     

bpy.ops.object.bake(type='DIFFUSE')

# FAST MEMORY STREAM: Allocate a contiguous C-array buffer for raw color bytes
# Each corner color contains 4 floats: (R, G, B, A)
print("Streaming shadow values to a raw memory buffer...")
shadow_buffer = array.array('f', [0.0]) * (num_loops * 4)
color_attr.data.foreach_get("color", shadow_buffer)


# --- BAKE PASS 2: AMBIENT OCCLUSION ---
print("Baking Pass 2: Ray-Traced Ambient Occlusion...")
scene.cycles.bake_type = 'AO'
scene.render.bake.target = 'VERTEX_COLORS'

# --- DYNAMIC AO DISTANCE CALCULATION ---
world_corners = [terrain_obj.matrix_world @ mathutils.Vector(corner) for corner in terrain_obj.bound_box]
x_co = [c.x for c in world_corners]
y_co = [c.y for c in world_corners]
z_co = [c.z for c in world_corners]

width_x = max(x_co) - min(x_co)
height_z = max(z_co) - min(z_co)

optimal_distance = height_z * 0.75
if optimal_distance == 0:
    optimal_distance = width_x * 0.05

scene.render.bake.max_ray_distance = optimal_distance

print(f"--- Dynamic Geometry Analysis ---")
print(f"Terrain Dimensions: Width={width_x:.2f}, Height/Relief={height_z:.2f}")
print(f"Calculated Optimal AO Ray Distance: {scene.render.bake.max_ray_distance:.2f}")

# Trigger AO bake (overwrites active layer data geometry layout)
bpy.ops.object.bake(type='AO')

# FAST MEMORY STREAM: Extract AO color data array layout
print("Streaming AO values to a raw memory buffer...")
ao_buffer = array.array('f', [0.0]) * (num_loops * 4)
color_attr.data.foreach_get("color", ao_buffer)


# --- LOW-LEVEL MEMORY BLITTING & MERGING ---
print("Merging data streams directly inside C-memory...")
# Pre-allocate a final combined flat buffer
packed_buffer = array.array('f', [0.0]) * (num_loops * 4)

# In a 4-channel layout, indices map as: 0=R, 1=G, 2=B, 3=A
# Slice the components at 4-byte boundaries across raw data strides
# This operates nearly instantaneously without running individual vertex loops
packed_buffer[0::4] = shadow_buffer[0::4] # Map raw shadow calculations to Red channel
packed_buffer[1::4] = ao_buffer[0::4]     # Map raw AO calculations to Green channel
packed_buffer[2::4] = array.array('f', [0.0]) * num_loops # Clear Blue channel
packed_buffer[3::4] = array.array('f', [1.0]) * num_loops # Set Alpha channel fully opaque

# Push the final structured layout straight back to the core mesh memory
print("Writing packed array back to the active mesh data structure...")
color_attr.data.foreach_set("color", packed_buffer)
mesh.update()


# --- CLEANUP MEMORY ---
del shadow_buffer
del ao_buffer
del packed_buffer


# --- EXPORT TO PLY ---
print("Exporting final packed PLY file...")
bpy.ops.wm.ply_export(
    filepath=EXPORT_PATH,
    export_selected_objects=True,
    export_normals=True,
    export_colors='SRGB',
    export_attributes=True
)

print(f"Success! Packed map base saved to: {EXPORT_PATH}")