import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import rasterio
from PyQt5 import QtWidgets, QtCore

# =====================================================================
# CONFIGURATION & PARAMETERS (ALL INITIAL VALUES)
# =====================================================================
DEM_TIF_PATH = "../data/wurl/USGS_13_n41w112_20260519.tif"

VIEWPORT_DOWNSAMPLE_STEP = 6  
PRODUCTION_DOWNSAMPLE_STEP = 4  
VIEWPORT_BUFFER = 0.15  

TARGET_LAT = 40.5754067
TARGET_LON = -111.7915164
PROJECTION_MODE = "sphere"  

CAMERA_DISTANCE = 75000.0   
CAMERA_ALTITUDE = 32000.0   

# Initial Tuning Parameters Dictionary (Radius Removed)
params = {
    "HORIZON_COMPRESSION": 0.75,
    "ALPHA": 3.5,
    "BETA": 2.2,
    "GLOBAL_Z_SCALE": 2.0,
    "HORIZON_LIFT_FACTOR": 3.0,
    "BLENDER_FILENAME": "warped.obj",
    "FLAT_FILENAME": "flat.obj"
}

# Documentation definitions for information popups
PARAM_DOCS = {
    "HORIZON_COMPRESSION": "Controls horizon compression (fisheye lens scaling).\n\nLower values squeeze the distant background elements closer to the center of your view plane, while larger values stretch them back out toward the edges.",
    "ALPHA": "Swoop Amplitude Exaggeration Factor.\n\nDirectly scales the steepness and height of the progressive tilt curve. Increasing Alpha makes the background warp upward aggressively like a dramatic painted panorama.",
    "BETA": "Swoop Acceleration Curve Exponential.\n\nDetermines how abruptly the terrain tilts upward. Low values make the transition gradual across the entire map; high values keep the foreground flat and suddenly shoot the background up like a wall.",
    "GLOBAL_Z_SCALE": "Vertical Exaggeration multiplier.\n\nPurely scales the raw elevation heights (Z-axis) of mountains and valleys uniformly across the entire DEM before any curved projections are calculated.",
    "HORIZON_LIFT_FACTOR": "Horizon lift amplifier.\n\nProgressively stretches and elevates terrain structures at the absolute furthest edges of your map field to make sure the horizon doesn't clip out of view."
}

# =====================================================================
# CORE GEOSPATIAL WARPING ENGINE
# =====================================================================
def run_berann_math(lons, lats, elevation, p):
    lat_to_meters = 111320.0
    lon_to_meters = 111320.0 * np.cos(np.radians(TARGET_LAT))

    x_metric = (lons - TARGET_LON) * lon_to_meters   
    y_metric = (lats - TARGET_LAT) * lat_to_meters   

    r_original = np.sqrt(x_metric**2 + y_metric**2)
    r_original[r_original == 0] = 1e-5
    
    # Automatically bound max radius to full extent of dataset bounds
    r_max = r_original.max()

    dome_scale_meters = r_max / p["HORIZON_COMPRESSION"]

    if PROJECTION_MODE.lower() == "sphere":
        d_radial = r_original / r_max
        
        horizon_scale_modifier = 1.0 + (d_radial ** 2.0) * (p["HORIZON_LIFT_FACTOR"] - 1.0)
        E_radial = 1.0 + p["ALPHA"] * (d_radial ** p["BETA"])
        elevation_deformed = elevation * E_radial * p["GLOBAL_Z_SCALE"] * horizon_scale_modifier
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
        horizon_scale_modifier = 1.0 + (d_linear ** 2.0) * (p["HORIZON_LIFT_FACTOR"] - 1.0)
        E_linear = 1.0 + p["ALPHA"] * (d_linear ** p["BETA"])
        elevation_deformed = elevation * p["GLOBAL_Z_SCALE"] * horizon_scale_modifier
        x_deformed = x_metric 
        y_offset = y_metric - y_min
        y_deformed = y_min + y_offset * (1.0 + (p["ALPHA"] / (p["BETA"] + 1)) * (d_linear ** p["BETA"]))
        
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
# PYQT MASTER WORKSPACE INTERFACE
# =====================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Advanced Berann Projection Control Studio")
        self.resize(1500, 950)

        self.camera_cache = {"position": None, "focal_point": None, "up": None, "view_angle": None, "window_size": None}
        self.updating_ui = False  

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setFixedWidth(360)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.sidebar)

        self.plotter_frame = QtInteractor(self)
        layout.addWidget(self.plotter_frame)

        self.load_base_dataset()
        self.build_ui_controls()
        self.init_3d_canvas()

    def load_base_dataset(self):
        print("Loading preview base dataset...")
        with rasterio.open(DEM_TIF_PATH) as proxy_src:
            out_height = int(proxy_src.height / VIEWPORT_DOWNSAMPLE_STEP)
            out_width = int(proxy_src.width / VIEWPORT_DOWNSAMPLE_STEP)
            self.elevation_preview = proxy_src.read(1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
            cols, rows = np.meshgrid(np.arange(0, proxy_src.width, VIEWPORT_DOWNSAMPLE_STEP)[:out_width], np.arange(0, proxy_src.height, VIEWPORT_DOWNSAMPLE_STEP)[:out_height])
            xs, ys = rasterio.transform.xy(proxy_src.transform, rows, cols)
            self.lons_preview = np.array(xs).reshape(out_height, out_width)
            self.lats_preview = np.array(ys).reshape(out_height, out_width)
            self.out_width = out_width
            self.out_height = out_height

    def create_parameter_row(self, key_id, display_label, min_v, max_v, default_v, resolution=100.0):
        """Creates a synced Row Widget container holding: Info Button, Title, Slider, and Nudge SpinBox."""
        container = QtWidgets.QWidget()
        row_lay = QtWidgets.QVBoxLayout(container)
        row_lay.setContentsMargins(0, 4, 0, 4)

        header_widget = QtWidgets.QWidget()
        header_lay = QtWidgets.QHBoxLayout(header_widget)
        header_lay.setContentsMargins(0, 0, 0, 0)

        info_btn = QtWidgets.QPushButton("?")
        info_btn.setFixedSize(20, 20)
        info_btn.setStyleSheet("border-radius: 10px; background-color: #4a5568; color: white; font-weight: bold;")
        info_btn.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, f"{display_label} Details", PARAM_DOCS[key_id]))
        header_lay.addWidget(info_btn)

        title_lbl = QtWidgets.QLabel(display_label)
        title_lbl.setStyleSheet("font-weight: bold;")
        header_lay.addWidget(title_lbl)
        header_lay.addStretch()

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setRange(min_v, max_v)
        spin_box.setValue(default_v)
        spin_box.setSingleStep(5.0 / resolution if min_v < 10 else 50.0)
        spin_box.setFixedWidth(80)
        header_lay.addWidget(spin_box)
        row_lay.addWidget(header_widget)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(min_v * resolution), int(max_v * resolution))
        slider.setValue(int(default_v * resolution))
        row_lay.addWidget(slider)

        def on_slider_moved(val):
            if self.updating_ui: return
            self.updating_ui = True
            spin_box.setValue(val / resolution)
            params[key_id] = val / resolution
            self.updating_ui = False
            self.update_mesh_geometry()

        def on_spin_changed(val):
            if self.updating_ui: return
            self.updating_ui = True
            slider.setValue(int(val * resolution))
            params[key_id] = val
            self.updating_ui = False
            self.update_mesh_geometry()

        slider.valueChanged.connect(on_slider_moved)
        spin_box.valueChanged.connect(on_spin_changed)

        self.sidebar.layout().addWidget(container)
        return slider, spin_box

    def build_ui_controls(self):
        """Constructs the side panel parameters panel (Radius elements omitted)."""
        self.c_sld, self.c_spn = self.create_parameter_row("HORIZON_COMPRESSION", "Horizon Compression", 0.1, 4.0, params["HORIZON_COMPRESSION"])
        self.a_sld, self.a_spn = self.create_parameter_row("ALPHA", "Alpha (Exaggeration Amplitude)", 0.0, 15.0, params["ALPHA"])
        self.b_sld, self.b_spn = self.create_parameter_row("BETA", "Beta (Curve Exponential)", 1.0, 6.0, params["BETA"])
        self.z_sld, self.z_spn = self.create_parameter_row("GLOBAL_Z_SCALE", "Global Z Scale Elevation", 0.1, 10.0, params["GLOBAL_Z_SCALE"])
        self.l_sld, self.l_spn = self.create_parameter_row("HORIZON_LIFT_FACTOR", "Horizon Lift Factor", 1.0, 12.0, params["HORIZON_LIFT_FACTOR"])
        
        self.sidebar.layout().addWidget(QtWidgets.QLabel("\n" + "="*38 + "\nEXPORT TARGET CONFIGURATION"))
        
        self.w_file_txt = QtWidgets.QLineEdit(params["BLENDER_FILENAME"])
        self.sidebar.layout().addWidget(QtWidgets.QLabel("Warped Mesh File Field Output Location:"))
        self.sidebar.layout().addWidget(self.w_file_txt)

        self.f_file_txt = QtWidgets.QLineEdit(params["FLAT_FILENAME"])
        self.sidebar.layout().addWidget(QtWidgets.QLabel("Flat Reference Mesh File Field Output Location:"))
        self.sidebar.layout().addWidget(self.f_file_txt)

        self.sidebar.layout().addWidget(QtWidgets.QLabel(""))
        
        self.bake_btn = QtWidgets.QPushButton("Bake Production High-Res OBJ")
        self.bake_btn.setStyleSheet("background-color: #1b4d3e; color: #ffffff; font-weight: bold; font-size: 11pt; padding: 10px; border-radius: 4px;")
        self.bake_btn.clicked.connect(self.execute_highres_production_bake)
        self.sidebar.layout().addWidget(self.bake_btn)

    def init_3d_canvas(self):
        X_w, Y_w, Z_w, r_orig = run_berann_math(self.lons_preview, self.lats_preview, self.elevation_preview, params)
        
        self.grid = pv.StructuredGrid()
        self.grid.points = np.column_stack((X_w.ravel(), Y_w.ravel(), Z_w.ravel()))
        self.grid.dimensions = (self.out_width, self.out_height, 1)
        self.preview_mesh = self.grid.extract_surface(algorithm='dataset_surface')

        self.plotter_frame.set_background("dimgray")
        self.plotter_frame.add_mesh(
            self.preview_mesh, 
            scalars=self.elevation_preview.ravel(), 
            cmap="terrain", 
            lighting=False, 
            show_scalar_bar=False
        )

        row, col = np.unravel_index(np.argmin(r_orig), r_orig.shape)
        target_focal = np.array([X_w[row, col], Y_w[row, col], Z_w[row, col]])

        initial_cam_pos = target_focal - np.array([0.0, CAMERA_DISTANCE, 0.0]) + np.array([0.0, 0.0, CAMERA_ALTITUDE])
        self.plotter_frame.camera.position = initial_cam_pos
        self.plotter_frame.camera.focal_point = target_focal
        self.plotter_frame.camera.up = np.array([0.0, 0.0, 1.0])

        self.plotter_frame.add_on_render_callback(self.render_callback)

    def render_callback(self, _p):
        self.camera_cache["position"] = np.array(self.plotter_frame.camera.position)
        self.camera_cache["focal_point"] = np.array(self.plotter_frame.camera.focal_point)
        self.camera_cache["up"] = np.array(self.plotter_frame.camera.up)
        self.camera_cache["view_angle"] = self.plotter_frame.camera.view_angle
        self.camera_cache["window_size"] = list(self.plotter_frame.window_size)

    def update_mesh_geometry(self):
        X_w, Y_w, Z_w, _ = run_berann_math(self.lons_preview, self.lats_preview, self.elevation_preview, params)
        self.preview_mesh.points = np.column_stack((X_w.ravel(), Y_w.ravel(), Z_w.ravel()))
        self.plotter_frame.render()

    def execute_highres_production_bake(self):
        params["BLENDER_FILENAME"] = self.w_file_txt.text()
        params["FLAT_FILENAME"] = self.f_file_txt.text()

        print("\n[BAKE INITIALIZED] Re-indexing full-resolution dataset files...")
        with rasterio.open(DEM_TIF_PATH) as prod_src:
            out_h = int(prod_src.height / PRODUCTION_DOWNSAMPLE_STEP)
            out_w = int(prod_src.width / PRODUCTION_DOWNSAMPLE_STEP)
            elevation_p = prod_src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            cols_p, rows_p = np.meshgrid(np.arange(0, prod_src.width, PRODUCTION_DOWNSAMPLE_STEP)[:out_w], np.arange(0, prod_src.height, PRODUCTION_DOWNSAMPLE_STEP)[:out_h])
            xs_p, ys_p = rasterio.transform.xy(prod_src.transform, rows_p, cols_p)
            lons_p = np.array(xs_p).reshape(out_h, out_w)
            lats_p = np.array(ys_p).reshape(out_h, out_w)

        X_p, Y_p, Z_p, r_p = run_berann_math(lons_p, lats_p, elevation_p, params)

        lon_to_meters = 111320.0 * np.cos(np.radians(TARGET_LAT))
        lat_to_meters = 111320.0
        X_flat_raw = (lons_p - TARGET_LON) * lon_to_meters
        Y_flat_raw = (lats_p - TARGET_LAT) * lat_to_meters
        Z_flat_raw = elevation_p * params["GLOBAL_Z_SCALE"]

        renderer = self.plotter_frame.renderer
        w_w, w_h = self.camera_cache["window_size"]

        pts_highres = np.column_stack((X_p.ravel(), Y_p.ravel(), Z_p.ravel()))
        homog_pts = np.hstack((pts_highres, np.ones((pts_highres.shape[0], 1))))

        view_matrix = np.array(renderer.GetActiveCamera().GetModelViewTransformMatrix().GetData()).reshape(4,4)
        proj_matrix = np.array(renderer.GetActiveCamera().GetProjectionTransformMatrix(w_w / w_h, -1, 1).GetData()).reshape(4,4)
        total_matrix = proj_matrix @ view_matrix

        clip_space = homog_pts @ total_matrix.T
        w_component = clip_space[:, 3][:, None]
        w_component[w_component == 0] = 1e-5
        ndc = clip_space[:, :3] / w_component

        buf = VIEWPORT_BUFFER 
        in_view_mask = (
            (ndc[:, 0] >= -(1.0 + buf)) & (ndc[:, 0] <= (1.0 + buf)) &  
            (ndc[:, 1] >= -(1.0 + buf)) & (ndc[:, 1] <= (1.0 + buf)) &  
            (ndc[:, 2] >= 0.0) & (ndc[:, 2] <= 1.0)                    
        ).reshape(out_h, out_w)

        rows_valid, cols_valid = np.where(in_view_mask)

        if len(rows_valid) > 0:
            r_start, r_end = rows_valid.min(), rows_valid.max() + 1
            c_start, c_end = cols_valid.min(), cols_valid.max() + 1
            X_cropped = X_p[r_start:r_end, c_start:c_end]
            Y_cropped = Y_p[r_start:r_end, c_start:c_end]
            Z_cropped = Z_p[r_start:r_end, c_start:c_end]
            X_f_cropped = X_flat_raw[r_start:r_end, c_start:c_end]
            Y_f_cropped = Y_flat_raw[r_start:r_end, c_start:c_end]
            Z_f_cropped = Z_flat_raw[r_start:r_end, c_start:c_end]
            crop_height, crop_width = X_cropped.shape
        else:
            print("Clipping Error: No points visible within camera window bounds."); return

        focal_world = self.camera_cache["focal_point"]
        cam_pos_world = self.camera_cache["position"]
        raw_distance = np.linalg.norm(focal_world - cam_pos_world)
        SCALE_FACTOR = 10.0 / raw_distance

        X_final = (X_cropped - focal_world[0]) * SCALE_FACTOR
        Y_final = (Y_cropped - focal_world[1]) * SCALE_FACTOR
        Z_final = (Z_cropped - focal_world[2]) * SCALE_FACTOR
        pts_blender_warped = np.column_stack((X_final.ravel(), Z_final.ravel(), -Y_final.ravel()))

        row_p, col_p = np.unravel_index(np.argmin(r_p), r_p.shape)
        target_elev_p = Z_flat_raw[row_p, col_p]

        X_f_final = X_f_cropped * SCALE_FACTOR
        Y_f_final = Y_f_cropped * SCALE_FACTOR
        Z_f_final = (Z_f_cropped - target_elev_p) * SCALE_FACTOR
        pts_blender_flat = np.column_stack((X_f_final.ravel(), Z_f_final.ravel(), -Y_f_final.ravel()))

        u = np.linspace(0, 1, crop_width)
        v = np.linspace(1, 0, crop_height)
        uu, vv = np.meshgrid(u, v)
        uv_pts = np.column_stack((uu.ravel(), vv.ravel()))

        r_idx = np.arange(crop_height - 1)[:, None]
        c_idx = np.arange(crop_width - 1)
        v1 = r_idx * crop_width + c_idx + 1
        v2 = v1 + 1
        v3 = (r_idx + 1) * crop_width + c_idx + 2
        v4 = v3 - 1
        faces = np.column_stack((v1.ravel(), v4.ravel(), v3.ravel(), v2.ravel()))
        faces_obj = np.column_stack((faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 3], faces[:, 3]))

        w_out = params["BLENDER_FILENAME"]
        f_out = params["FLAT_FILENAME"]
        
        print(f"Saving high resolution mesh to: {w_out}...")
        with open(w_out, 'w', encoding='utf-8') as f:
            f.write("# Clean Production Warped Mesh\n")
            np.savetxt(f, pts_blender_warped, fmt="v %.5f %.5f %.5f")
            np.savetxt(f, uv_pts, fmt="vt %.5f %.5f")
            np.savetxt(f, faces_obj, fmt="f %d/%d %d/%d %d/%d %d/%d")

        print(f"Saving companion unwarped mesh to: {f_out}...")
        with open(f_out, 'w', encoding='utf-8') as f:
            f.write("# Clean Production Companion Flat Map\n")
            np.savetxt(f, pts_blender_flat, fmt="v %.5f %.5f %.5f")
            np.savetxt(f, uv_pts, fmt="vt %.5f %.5f")
            np.savetxt(f, faces_obj, fmt="f %d/%d %d/%d %d/%d %d/%d")
            
        print("=== MESH PACKAGING BAKE SUCCESSFUL ===")

    def closeEvent(self, event):
        self.plotter_frame.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())