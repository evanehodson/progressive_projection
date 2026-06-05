import numpy as np
import pyvista as pv
import rasterio

# =====================================================================
# CONFIGURATION & PARAMETERS (ALL UP TOP)
# =====================================================================
DEM_TIF_PATH = "data/wurl/USGS_13_n41w112_20260519.tif"
DOWNSAMPLE_STEP = 2  

# View Center Reference
TARGET_LAT = 40.5754067
TARGET_LON = -111.7915164

# --- PROJECTION PARAMETERS ---
PROJECTION_MODE = "sphere"  
HORIZON_COMPRESSION = 0.75  

# --- PROGRESSIVE EXAGGERATION CONTROLS ---
ALPHA = 3.5          
BETA = 2.2           
GLOBAL_Z_SCALE = 2.0 # Slightly increased to give sharper vertical definition

# --- BERANN HORIZON LIFT OVERRIDE ---
HORIZON_LIFT_FACTOR = 2.2  

# --- PRODUCTION OUTPUT SETTINGS ---
OUTPUT_RESOLUTION = (4000, 3000)  
OUTPUT_FILENAME = "cottonwood_berann_hillshade.tif"

# --- THEATRICAL LIGHTING RIG CONTROLS (RE-BALANCED CONTRAST) ---
SUN_INTENSITY = 1.45   # Powerful sun to pull vibrant whites out of the gray mesh
RIM_INTENSITY = 0.65   # Strong rim definition for peak separations
FILL_INTENSITY = 0.05  # Reduced to near-zero to keep shadows crisp and dark

# Base Camera Starting Frame Coordinates
CAMERA_DISTANCE = 75000.0   
CAMERA_ALTITUDE = 32000.0   

# =====================================================================
# STEP 1: LOAD DEM DATA WITH RESAMPLING
# =====================================================================
print("Loading elevation dataset...")
with rasterio.open(DEM_TIF_PATH) as src:
    out_height = int(src.height / DOWNSAMPLE_STEP)
    out_width = int(src.width / DOWNSAMPLE_STEP)
    
    elevation = src.read(1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.cubic)
    cols, rows = np.meshgrid(np.arange(0, src.width, DOWNSAMPLE_STEP)[:out_width], np.arange(0, src.height, DOWNSAMPLE_STEP)[:out_height])
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons = np.array(xs).reshape(out_height, out_width)
    lats = np.array(ys).reshape(out_height, out_width)

# =====================================================================
# STEP 2: METRIC GEOMETRY TRANSFORMATIONS
# =====================================================================
lat_to_meters = 111320.0
lon_to_meters = 111320.0 * np.cos(np.radians(TARGET_LAT))

x_metric = (lons - TARGET_LON) * lon_to_meters   
y_metric = (lats - TARGET_LAT) * lat_to_meters   

r_original = np.sqrt(x_metric**2 + y_metric**2)
r_original[r_original == 0] = 1e-5
r_max = r_original.max()

dome_scale_meters = r_max / HORIZON_COMPRESSION

if PROJECTION_MODE.lower() == "sphere":
    d_radial = r_original / r_max
    horizon_scale_modifier = 1.0 + (d_radial ** 2.0) * (HORIZON_LIFT_FACTOR - 1.0)
    E_radial = 1.0 + ALPHA * (d_radial ** BETA)
    elevation_deformed = elevation * E_radial * GLOBAL_Z_SCALE * horizon_scale_modifier
    r_deformed = r_original * E_radial
    theta = np.arctan2(y_metric, x_metric)
    
    phi = r_deformed / dome_scale_meters
    X_world = dome_scale_meters * np.sin(phi) * np.cos(theta)
    Y_world = dome_scale_meters * np.sin(phi) * np.sin(theta)
    Z_base = dome_scale_meters * (np.cos(phi) - 1.0)
    
    norm_x = np.sin(phi) * np.cos(theta)
    norm_y = np.sin(phi) * np.sin(theta)
    norm_z = np.cos(phi)
else:
    y_min, y_max = y_metric.min(), y_metric.max()
    d_linear = (y_metric - y_min) / (y_max - y_min)
    horizon_scale_modifier = 1.0 + (d_linear ** 2.0) * (HORIZON_LIFT_FACTOR - 1.0)
    E_linear = 1.0 + ALPHA * (d_linear ** BETA)
    elevation_deformed = elevation * E_linear * GLOBAL_Z_SCALE * horizon_scale_modifier
    x_deformed = x_metric 
    y_offset = y_metric - y_min
    y_deformed = y_min + y_offset * (1.0 + (ALPHA / (BETA + 1)) * (d_linear ** BETA))
    
    phi_y = (y_deformed - y_min) / dome_scale_meters
    X_world = x_deformed
    Y_world = y_min + dome_scale_meters * np.sin(phi_y)
    Z_base = dome_scale_meters * (np.cos(phi_y) - 1.0)
    
    norm_x = np.zeros_like(X_world)
    norm_y = np.sin(phi_y)
    norm_z = np.cos(phi_y)

X_world += elevation_deformed * norm_x
Y_world += elevation_deformed * norm_y
Z_world = Z_base + elevation_deformed * norm_z

# =====================================================================
# STEP 3: MESH PACKAGING & SHARP RIDGE PROTECTION
# =====================================================================
print("Generating landscape surfaces...")
grid = pv.StructuredGrid()
grid.points = np.column_stack((X_world.ravel(), Y_world.ravel(), Z_world.ravel()))
grid.dimensions = (out_width, out_height, 1)

grid.point_data["Elevation"] = elevation.ravel()
surface_mesh = grid.extract_surface(algorithm='dataset_surface')

print("Applying feature-preserving smoothing...")
# Using fewer iterations and a strict edge_angle threshold keeps the ridge peaks sharp 
# while smoothing out step artifacts along the valleys.
surface_mesh = surface_mesh.smooth(n_iter=15, relaxation_factor=0.03, edge_angle=35.0, boundary_smoothing=False)

print("Calculating vertex shading vectors...")
surface_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

target_row, target_col = np.unravel_index(np.argmin(r_original), r_original.shape)
target_focal = np.array([X_world[target_row, target_col], Y_world[target_row, target_col], Z_world[target_row, target_col]])

# =====================================================================
# STEP 4: INTERACTIVE SETUP & MATERIAL CONFIGURATION
# =====================================================================
print("\n=== Phase 1: Interactive View Setup ===")
plotter = pv.Plotter()
plotter.remove_all_lights()

plotter.add_mesh(
    surface_mesh, 
    color="whitesmoke",  # Brighter base color gives greater dynamic range for shadows
    show_edges=False, 
    lighting=True, 
    pbr=True,            
    roughness=0.60,      # Slightly lowered to help catching clean light highlights
    metallic=0.0,       
    specular=0.25,       # Increased specular creates high-contrast highlights along ridge crests
    ambient=0.0          # Pure zero ambient lets shadows drop completely into deep blacks
)

def setup_berann_lighting(plotter_instance, cam_pos, focal_pt):
    plotter_instance.remove_all_lights()
    
    cam_vec = cam_pos - focal_pt
    cam_dist = np.linalg.norm(cam_vec)
    cam_dir_normalized = cam_vec / (cam_dist if cam_dist > 0 else 1.0)
    
    # Light 1: The Berann Key Light (Angled off-shoulder and lifted higher)
    # Moving it 25 degrees off the camera axis introduces deep shadows along the flanks
    cos_rot = np.cos(np.radians(25.0))
    sin_rot = np.sin(np.radians(25.0))
    rotated_dir = np.array([
        cam_dir_normalized[0] * cos_rot - cam_dir_normalized[1] * sin_rot,
        cam_dir_normalized[0] * sin_rot + cam_dir_normalized[1] * cos_rot,
        cam_dir_normalized[2]
    ])
    
    sun_pos = focal_pt + (rotated_dir * 250000.0)
    sun_pos[2] = focal_pt[2] + 65000.0  # Kept higher to cast crisp, legible down-canyon shadows
    sun_light = pv.Light(position=sun_pos, focal_point=focal_pt, intensity=SUN_INTENSITY, light_type='scenelight')
    plotter_instance.add_light(sun_light)
    
    # Light 2: Rim-Separator Counter Light
    rim_pos = focal_pt - (cam_dir_normalized * 250000.0)
    rim_pos[2] = focal_pt[2] + 35000.0  
    rim_light = pv.Light(position=rim_pos, focal_point=focal_pt, intensity=RIM_INTENSITY, light_type='scenelight')
    plotter_instance.add_light(rim_light)
    
    # Light 3: Soft Overhead Sky Fill
    fill_light = pv.Light(position=focal_pt + [0, 0, 300000.0], focal_point=focal_pt, intensity=FILL_INTENSITY, light_type='scenelight')
    plotter_instance.add_light(fill_light)

initial_cam_pos = target_focal - np.array([0.0, CAMERA_DISTANCE, 0.0]) + np.array([0.0, 0.0, CAMERA_ALTITUDE])
plotter.camera.position = initial_cam_pos
plotter.camera.focal_point = target_focal
plotter.camera.up = np.array([0.0, 0.0, 1.0])
plotter.set_background("white")

setup_berann_lighting(plotter, initial_cam_pos, target_focal)

def structural_light_callback(_plotter):
    setup_berann_lighting(plotter, np.array(plotter.camera.position), np.array(plotter.camera.focal_point))

plotter.add_on_render_callback(structural_light_callback)

# Tighter radius and increased bias concentrates ambient occlusion into deep, hand-carved valley cuts
plotter.enable_ssao(radius=3500.0, bias=0.55, blur=True)

plotter.show()

# =====================================================================
# STEP 5: HIGH-RES PRODUCTION RENDER OVERRIDE
# =====================================================================
DYNAMIC_CAM_POSITION = list(plotter.camera.position)
DYNAMIC_CAM_FOCAL    = list(plotter.camera.focal_point)
DYNAMIC_CAM_UP       = list(plotter.camera.up)

print("\n=== Phase 2: Generating Production Output Image ===")

p_out = pv.Plotter(window_size=OUTPUT_RESOLUTION, off_screen=True)
p_out.add_mesh(
    surface_mesh, 
    color="whitesmoke", 
    show_edges=False, 
    lighting=True, 
    pbr=True, 
    roughness=0.60, 
    metallic=0.0,
    specular=0.25,
    ambient=0.0
)

p_out.enable_ssao(radius=3500.0, bias=0.55, blur=True)
p_out.set_background("white")

p_out.camera.position = DYNAMIC_CAM_POSITION
p_out.camera.focal_point = DYNAMIC_CAM_FOCAL
p_out.camera.up = DYNAMIC_CAM_UP
p_out.camera.view_angle = 30.0

setup_berann_lighting(p_out, np.array(DYNAMIC_CAM_POSITION), np.array(DYNAMIC_CAM_FOCAL))

p_out.screenshot(OUTPUT_FILENAME)
p_out.close()

print(f"\n=== PRODUCTION ASSET SAVED: {OUTPUT_FILENAME} ===")