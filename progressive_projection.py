import numpy as np
import pyvista as pv
import rasterio

# =====================================================================
# CONFIGURATION & PARAMETERS (ALL UP TOP)
# =====================================================================
DEM_TIF_PATH = "../data/wurl/USGS_13_n41w112_20260519.tif"

# --- PYVISTA VIEWPORT ACCELERATION ---
VIEWPORT_DOWNSAMPLE_STEP = 6  

# --- OPTIMIZED BLENDER PRODUCTION RESOLUTION ---
PRODUCTION_DOWNSAMPLE_STEP = 4  

# Padding added outside the visible camera frame borders (0.15 = 15% extra padding)
VIEWPORT_BUFFER = 0.15  

# View Center Reference
TARGET_LAT = 40.5754067
TARGET_LON = -111.7915164

# --- PROJECTION PARAMETERS ---
PROJECTION_MODE = "sphere"  
HORIZON_COMPRESSION = 0.75  

# --- PROGRESSIVE EXAGGERATION CONTROLS ---
ALPHA = 3.5          
BETA = 2.2           
GLOBAL_Z_SCALE = 2.0 

# --- BERANN HORIZON LIFT OVERRIDE ---
HORIZON_LIFT_FACTOR = 2.2  

# --- PRODUCTION BLENDER EXPORT SETTINGS ---
BLENDER_EXPORT_FILENAME = "cottonwood_warped_for_blender.obj"

# --- LITE VIEWPORT LIGHTING RIG CONTROLS ---
SUN_INTENSITY = 1.45   
RIM_INTENSITY = 0.65   
FILL_INTENSITY = 0.05  

CAMERA_DISTANCE = 75000.0   
CAMERA_ALTITUDE = 32000.0   

# =====================================================================
# CORE GEOSPATIAL WARPING ENGINE
# =====================================================================
def run_berann_math(lons, lats, elevation):
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
    
    return X_world, Y_world, Z_world, r_original

# =====================================================================
# PHASE 1: LIGHTWEIGHT VIEWPORT SETUP
# =====================================================================
print("\n=== Phase 1: Interactive View Setup ===")
with rasterio.open(DEM_TIF_PATH) as proxy_src:
    out_height = int(proxy_src.height / VIEWPORT_DOWNSAMPLE_STEP)
    out_width = int(proxy_src.width / VIEWPORT_DOWNSAMPLE_STEP)
    elevation = proxy_src.read(1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
    cols, rows = np.meshgrid(np.arange(0, proxy_src.width, VIEWPORT_DOWNSAMPLE_STEP)[:out_width], np.arange(0, proxy_src.height, VIEWPORT_DOWNSAMPLE_STEP)[:out_height])
    xs, ys = rasterio.transform.xy(proxy_src.transform, rows, cols)
    lons = np.array(xs).reshape(out_height, out_width)
    lats = np.array(ys).reshape(out_height, out_width)

X_world, Y_world, Z_world, r_original = run_berann_math(lons, lats, elevation)
row, col = np.unravel_index(np.argmin(r_original), r_original.shape)
target_focal = np.array([X_world[row, col], Y_world[row, col], Z_world[row, col]])

grid = pv.StructuredGrid()
grid.points = np.column_stack((X_world.ravel(), Y_world.ravel(), Z_world.ravel()))
grid.dimensions = (out_width, out_height, 1)
preview_mesh = grid.extract_surface(algorithm='dataset_surface')

plotter = pv.Plotter()
plotter.remove_all_lights()
preview_mesh = preview_mesh.smooth(n_iter=10, relaxation_factor=0.03, edge_angle=35.0, boundary_smoothing=False)
preview_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
plotter.add_mesh(preview_mesh, color="whitesmoke", show_edges=False, lighting=True, pbr=True, roughness=0.60, specular=0.25)

camera_cache = {
    "position": None,
    "focal_point": None,
    "up": None,
    "view_angle": None,
    "window_size": None
}

def setup_berann_lighting(plotter_instance, cam_pos, focal_pt):
    plotter_instance.remove_all_lights()
    cam_vec = cam_pos - focal_pt
    cam_dir_normalized = cam_vec / (np.linalg.norm(cam_vec) if np.linalg.norm(cam_vec) > 0 else 1.0)
    cos_rot, sin_rot = np.cos(np.radians(25.0)), np.sin(np.radians(25.0))
    rotated_dir = np.array([cam_dir_normalized[0]*cos_rot - cam_dir_normalized[1]*sin_rot, cam_dir_normalized[0]*sin_rot + cam_dir_normalized[1]*cos_rot, cam_dir_normalized[2]])
    plotter_instance.add_light(pv.Light(position=focal_pt + (rotated_dir * 250000.0) + [0,0,65000.0], focal_point=focal_pt, intensity=SUN_INTENSITY))
    plotter_instance.add_light(pv.Light(position=focal_pt - (cam_dir_normalized * 250000.0) + [0,0,35000.0], focal_point=focal_pt, intensity=RIM_INTENSITY))
    plotter_instance.add_light(pv.Light(position=focal_pt + [0, 0, 300000.0], focal_point=focal_pt, intensity=FILL_INTENSITY))

def render_callback(_p):
    camera_cache["position"] = np.array(plotter.camera.position)
    camera_cache["focal_point"] = np.array(plotter.camera.focal_point)
    camera_cache["up"] = np.array(plotter.camera.up)
    camera_cache["view_angle"] = plotter.camera.view_angle
    camera_cache["window_size"] = list(plotter.window_size)
    setup_berann_lighting(plotter, camera_cache["position"], camera_cache["focal_point"])

initial_cam_pos = target_focal - np.array([0.0, CAMERA_DISTANCE, 0.0]) + np.array([0.0, 0.0, CAMERA_ALTITUDE])
plotter.camera.position = initial_cam_pos
plotter.camera.focal_point = target_focal
plotter.camera.up = np.array([0.0, 0.0, 1.0])
plotter.set_background("white")

plotter.enable_ssao(radius=3500.0, bias=0.55)
render_callback(plotter)
plotter.add_on_render_callback(render_callback)
plotter.show()

# =====================================================================
# PHASE 2: FRUSTUM CROP & COORDINATE REMAPPING
# =====================================================================
print("\n=== Phase 2: Processing High-Resolution Dataset ===")

cam_pos_world = camera_cache["position"]
focal_world   = camera_cache["focal_point"]

fov_v   = camera_cache["view_angle"]
aspect  = camera_cache["window_size"][0] / camera_cache["window_size"][1]
fov_h   = np.degrees(2.0 * np.arctan(np.tan(np.radians(fov_v * 0.5)) * aspect))

view_dir  = focal_world - cam_pos_world
view_dir /= np.linalg.norm(view_dir)
up_vec    = camera_cache["up"]
up_vec   -= (up_vec @ view_dir) * view_dir
up_vec   /= np.linalg.norm(up_vec)
right_vec = np.cross(view_dir, up_vec)

pts      = preview_mesh.points
to_pts   = pts - cam_pos_world
depth    = to_pts @ view_dir
up_proj  = to_pts @ up_vec
rt_proj  = to_pts @ right_vec

buf    = 1.0 + VIEWPORT_BUFFER
half_h = np.tan(np.radians(fov_v * 0.5)) * depth
half_w = np.tan(np.radians(fov_h * 0.5)) * depth

in_frustum = (
    (depth > 0) &
    (up_proj >= -half_h * buf) & (up_proj <= half_h * buf) &
    (rt_proj >= -half_w * buf) & (rt_proj <= half_w * buf)
)
visible_pts = pts[in_frustum]

del preview_mesh
plotter.close()
del plotter

if visible_pts.shape[0] == 0:
    raise RuntimeError(
        "Frustum crop returned 0 points. "
        "Check CAMERA_DISTANCE / CAMERA_ALTITUDE, or widen VIEWPORT_BUFFER."
    )

v_x, v_y = visible_pts[:, 0], visible_pts[:, 1]
buffered_x_min, buffered_x_max = v_x.min(), v_x.max()
buffered_y_min, buffered_y_max = v_y.min(), v_y.max()

print("Loading high-resolution production dataset...")
with rasterio.open(DEM_TIF_PATH) as prod_src:
    out_height = int(prod_src.height / PRODUCTION_DOWNSAMPLE_STEP)
    out_width = int(prod_src.width / PRODUCTION_DOWNSAMPLE_STEP)
    elevation = prod_src.read(1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
    cols, rows = np.meshgrid(np.arange(0, prod_src.width, PRODUCTION_DOWNSAMPLE_STEP)[:out_width], np.arange(0, prod_src.height, PRODUCTION_DOWNSAMPLE_STEP)[:out_height])
    xs, ys = rasterio.transform.xy(prod_src.transform, rows, cols)
    lons = np.array(xs).reshape(out_height, out_width)
    lats = np.array(ys).reshape(out_height, out_width)

X_p, Y_p, Z_p, _ = run_berann_math(lons, lats, elevation)

crop_mask = (X_p >= buffered_x_min) & (X_p <= buffered_x_max) & (Y_p >= buffered_y_min) & (Y_p <= buffered_y_max)
rows_valid, cols_valid = np.where(crop_mask)

if len(rows_valid) > 0:
    r_start, r_end = rows_valid.min(), rows_valid.max() + 1
    c_start, c_end = cols_valid.min(), cols_valid.max() + 1
    
    X_cropped = X_p[r_start:r_end, c_start:c_end]
    Y_cropped = Y_p[r_start:r_end, c_start:c_end]
    Z_cropped = Z_p[r_start:r_end, c_start:c_end]
    crop_height, crop_width = X_cropped.shape
else:
    X_cropped, Y_cropped, Z_cropped = X_p, Y_p, Z_p
    crop_height, crop_width = out_height, out_width

# Center geometry around focus target
X_centered = X_cropped - focal_world[0]
Y_centered = Y_cropped - focal_world[1]
Z_centered = Z_cropped - focal_world[2]

# Standardize workspace units relative to PyVista real-world system range
raw_distance = np.linalg.norm(focal_world - cam_pos_world)
SCALE_FACTOR = 10.0 / raw_distance

X_final = X_centered * SCALE_FACTOR
Y_final = Y_centered * SCALE_FACTOR
Z_final = Z_centered * SCALE_FACTOR

# Axis swap to Y-Up convention (X, Z, -Y)
pts_blender = np.column_stack((X_final.ravel(), Z_final.ravel(), -Y_final.ravel()))

print(f"Writing {BLENDER_EXPORT_FILENAME}...")
with open(BLENDER_EXPORT_FILENAME, 'w', encoding='utf-8') as f:
    f.write("# Berann Warped Landscape\n")
    f.write("# Blender import: File > Import > Wavefront (.obj)\n")
    f.write("#   Forward Axis: -Z   |   Up Axis: Y\n")
    f.write("# (Set these explicitly; they differ from Blender 4.x defaults)\n")
    np.savetxt(f, pts_blender, fmt="v %.5f %.5f %.5f")
    
    r_idx = np.arange(crop_height - 1)[:, None]
    c_idx = np.arange(crop_width - 1)
    
    v1 = r_idx * crop_width + c_idx + 1
    v2 = v1 + 1
    v3 = (r_idx + 1) * crop_width + c_idx + 2
    v4 = v3 - 1
    
    # CCW winding adjustment from above to force normals upward towards your camera viewpoint
    faces = np.column_stack((v1.ravel(), v4.ravel(), v3.ravel(), v2.ravel()))
    np.savetxt(f, faces, fmt="f %d %d %d %d")

cam_offset = (cam_pos_world - focal_world) * SCALE_FACTOR

print("\n=== COMPLETE ===")
print("\n--- BLENDER CAMERA SETUP ---")
print("Import Settings: Forward Axis = -Z  |  Up Axis = Y")
print(f"Camera Location: X={cam_offset[0]:.4f}  Y={cam_offset[1]:.4f}  Z={cam_offset[2]:.4f}")
print("Track To:        target Empty at (0,0,0)  |  Track Axis: -Z  |  Up: Y")