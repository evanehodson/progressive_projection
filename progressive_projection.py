import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import rasterio
from PyQt5 import QtWidgets, QtCore, QtGui

DEM_TIF_PATH = "../data/wurl/USGS_13_n41w112_20260519.tif"
VIEWPORT_DOWNSAMPLE_STEP = 6  
PRODUCTION_DOWNSAMPLE_STEP = 4  
VIEWPORT_BUFFER = 0.05  

TARGET_LAT = 40.5754067
TARGET_LON = -111.7915164
CAMERA_DISTANCE = 75000.0   
CAMERA_ALTITUDE = 32000.0   

params = {
    "GLOBAL_Z": 2.5,
    "HORIZON_AREA_MAG": 1.0,    
    "HORIZON_RELIEF_MAG": 1.0,  
    "BEZIER_P1": [0.33, 0.0],   
    "BEZIER_P2": [0.66, -0.4],  
    "BEZIER_P3": [1.0, -0.6],   
    "BLENDER_FILENAME": "warped.obj",
    "FLAT_FILENAME": "flat.obj"
}

PARAM_DOCS = {
    "GLOBAL_Z": "Pre-distortion structural relief multiplier.",
    "HORIZON_AREA_MAG": "Horizontal cross-sectional stretching factor at the horizon bounds (X/Y axis expansion).",
    "HORIZON_RELIEF_MAG": "Vertical height magnification factor applied exclusively to the grounded relief near the horizon (Z axis expansion)."
}

class BezierCurveEditor(QtWidgets.QWidget):
    curveChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(340, 260)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.xmin, self.xmax = -0.05, 1.05
        self.ymin, self.ymax = -1.55, 0.05
        self.selected_point = None
        self.hover_point = None

    def to_screen(self, x, y):
        return QtCore.QPoint(
            int(((x - self.xmin) / (self.xmax - self.xmin)) * self.width()),
            int(((self.ymax - y) / (self.ymax - self.ymin)) * self.height())
        )

    def to_graph(self, sx, sy):
        return (
            self.xmin + (sx / self.width()) * (self.xmax - self.xmin),
            self.ymax - (sy / self.height()) * (self.ymax - self.ymin)
        )

    def get_points_dict(self):
        return {
            "P0": np.array([0.0, 0.0]),
            "P1": np.array(params["BEZIER_P1"]),
            "P2": np.array(params["BEZIER_P2"]),
            "P3": np.array(params["BEZIER_P3"])
        }

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#2d3748"))

        painter.setPen(QtGui.QPen(QtGui.QColor("#4a5568"), 1, QtCore.Qt.DashLine))
        for x_grid in np.linspace(0, 1, 5):
            painter.drawLine(self.to_screen(x_grid, self.ymin), self.to_screen(x_grid, self.ymax))
        for y_grid in np.linspace(-1.5, 0, 4):
            painter.drawLine(self.to_screen(self.xmin, y_grid), self.to_screen(self.xmax, y_grid))

        pts = self.get_points_dict()
        p0_s, p1_s = self.to_screen(*pts["P0"]), self.to_screen(*pts["P1"])
        p2_s, p3_s = self.to_screen(*pts["P2"]), self.to_screen(*pts["P3"])

        painter.setPen(QtGui.QPen(QtGui.QColor("#a0aec0"), 1.5, QtCore.Qt.SolidLine))
        painter.drawLine(p0_s, p1_s)
        painter.drawLine(p3_s, p2_s)

        painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 3, QtCore.Qt.SolidLine))
        t_arr = np.linspace(0, 1, 150)
        mt = 1.0 - t_arr
        x_vals = (mt**3)*pts["P0"][0] + 3*(mt**2)*t_arr*pts["P1"][0] + 3*mt*(t_arr**2)*pts["P2"][0] + (t_arr**3)*pts["P3"][0]
        y_vals = (mt**3)*pts["P0"][1] + 3*(mt**2)*t_arr*pts["P1"][1] + 3*mt*(t_arr**2)*pts["P2"][1] + (t_arr**3)*pts["P3"][1]

        poly_path = QtGui.QPolygonF([QtCore.QPointF(self.to_screen(x, y)) for x, y in zip(x_vals, y_vals)])
        painter.drawPolyline(poly_path)

        colors = {"P1": QtGui.QColor("#f59e0b"), "P2": QtGui.QColor("#ec4899"), "P3": QtGui.QColor("#ef4444")}
        for name, scr_pt in [("P1", p1_s), ("P2", p2_s), ("P3", p3_s)]:
            painter.setBrush(colors[name])
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            radius = 7 if self.hover_point == name else 5
            painter.drawEllipse(scr_pt, radius, radius)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            pts = self.get_points_dict()
            for name in ["P1", "P2", "P3"]:
                if (event.pos() - self.to_screen(*pts[name])).manhattanLength() < 12:
                    self.selected_point = name
                    self.update()
                    break

    def mouseMoveEvent(self, event):
        pts = self.get_points_dict()
        if not self.selected_point:
            old_hover = self.hover_point
            self.hover_point = None
            for name in ["P1", "P2", "P3"]:
                if (event.pos() - self.to_screen(*pts[name])).manhattanLength() < 12:
                    self.hover_point = name
                    break
            if old_hover != self.hover_point:
                self.update()
            return

        gx, gy = self.to_graph(event.x(), event.y())
        gx, gy = np.clip(gx, 0.0, 1.0), np.clip(gy, -1.5, 0.0)

        if self.selected_point == "P1": params["BEZIER_P1"] = [gx, gy]
        elif self.selected_point == "P2": params["BEZIER_P2"] = [gx, gy]
        elif self.selected_point == "P3": params["BEZIER_P3"] = [1.0, gy]

        self.update()
        self.curveChanged.emit()

    def mouseReleaseEvent(self, event):
        self.selected_point = None
        self.update()


def run_berann_math(lons, lats, elevation, p):
    lat_to_meters = 111320.0
    lon_to_meters = 111320.0 * np.cos(np.radians(TARGET_LAT))

    x_metric = (lons - TARGET_LON) * lon_to_meters   
    y_metric = (lats - TARGET_LAT) * lat_to_meters   

    r_original = np.sqrt(x_metric**2 + y_metric**2)
    r_max = r_original.max()
    d = r_original / r_max

    elevation_scaled = elevation * p["GLOBAL_Z"]
    min_elev_scaled = elevation_scaled.min()
    elevation_grounded = elevation_scaled - min_elev_scaled

    F_relief = 1.0 + (p["HORIZON_RELIEF_MAG"] - 1.0) * (d ** 2)

    flat_elev = elevation.ravel()
    sorted_elev = np.sort(flat_elev)
    idx_80 = int(0.80 * (len(sorted_elev) - 1))
    elev_80 = sorted_elev[idx_80]
    elev_max = sorted_elev[-1]
    
    percentile_rank = (elevation - elev_80) / np.where((elev_max - elev_80) == 0, 1e-5, (elev_max - elev_80))
    percentile_rank = np.clip(percentile_rank, 0.0, 1.0)
    
    f_h = np.where(elevation < elev_80, 0.0, (percentile_rank ** 4) * (3000.0 * (p["HORIZON_RELIEF_MAG"] - 1.0)))
    total_topographic_relief = (elevation_grounded * F_relief) + (d * f_h) + min_elev_scaled

    P0, P1, P2, P3 = np.array([0.0, 0.0]), np.array(p["BEZIER_P1"]), np.array(p["BEZIER_P2"]), np.array(p["BEZIER_P3"])
    t = np.linspace(0, 1, 1000)
    mt = 1.0 - t
    X_curve = (mt**3)*P0[0] + 3*(mt**2)*t*P1[0] + 3*mt*(t**2)*P2[0] + (t**3)*P3[0]
    Y_curve = (mt**3)*P0[1] + 3*(mt**2)*t*P1[1] + 3*mt*(t**2)*P2[1] + (t**3)*P3[1]
    
    dX_dt = -3*(mt**2)*P0[0] + 3*(1.0 - 4.0*t + 3.0*t**2)*P1[0] + 3*(2.0*t - 3.0*t**2)*P2[0] + 3*(t**2)*P3[0]
    dY_dt = -3*(mt**2)*P0[1] + 3*(1.0 - 4.0*t + 3.0*t**2)*P1[1] + 3*(2.0*t - 3.0*t**2)*P2[1] + 3*(t**2)*P3[1]
    dY_dX_curve = dY_dt / np.where(dX_dt == 0, 1e-8, dX_dt)

    d_warped_raw = d * (1.0 + (p["HORIZON_AREA_MAG"] - 1.0) * (d ** 2))
    
    flat_d = d.ravel()
    sort_idx = np.argsort(flat_d)
    inverse_idx = np.argsort(sort_idx)
    
    sorted_dw_monotonized = np.maximum.accumulate(d_warped_raw.ravel()[sort_idx])
    d_warped = np.clip(sorted_dw_monotonized[inverse_idx].reshape(d.shape), 0.0, 1.0)

    Z_base = np.interp(d_warped, X_curve, Y_curve) * r_max
    dz_dr = np.interp(d_warped, X_curve, dY_dX_curve)

    scale_warped = np.where(d == 0, 1.0, d_warped / d)
    x_warped_base = x_metric * scale_warped
    y_warped_base = y_metric * scale_warped
    r_warped_safe = np.where(r_original * scale_warped == 0, 1e-5, r_original * scale_warped)

    dz_dx = np.where(r_original == 0, 0.0, dz_dr * (x_warped_base / r_warped_safe))
    dz_dy = np.where(r_original == 0, 0.0, dz_dr * (y_warped_base / r_warped_safe))

    nx, ny, nz = -dz_dx, -dz_dy, np.ones_like(Z_base)
    norm_len = np.sqrt(nx**2 + ny**2 + nz**2)

    return (
        x_warped_base + total_topographic_relief * (nx / norm_len),
        y_warped_base + total_topographic_relief * (ny / norm_len),
        Z_base + total_topographic_relief * (nz / norm_len),
        r_original,
        dz_dr
    )


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
        with rasterio.open(DEM_TIF_PATH) as proxy_src:
            self.out_height = int(proxy_src.height / VIEWPORT_DOWNSAMPLE_STEP)
            self.out_width = int(proxy_src.width / VIEWPORT_DOWNSAMPLE_STEP)
            self.elevation_preview = proxy_src.read(1, out_shape=(self.out_height, self.out_width), resampling=rasterio.enums.Resampling.bilinear)
            cols, rows = np.meshgrid(np.arange(0, proxy_src.width, VIEWPORT_DOWNSAMPLE_STEP)[:self.out_width], np.arange(0, proxy_src.height, VIEWPORT_DOWNSAMPLE_STEP)[:self.out_height])
            xs, ys = rasterio.transform.xy(proxy_src.transform, rows, cols)
            self.lons_preview = np.array(xs).reshape(self.out_height, self.out_width)
            self.lats_preview = np.array(ys).reshape(self.out_height, self.out_width)

    def create_parameter_row(self, key_id, display_label, min_v, max_v, default_v, resolution=100.0):
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
        spin_box.setSingleStep(0.1 if max_v <= 10.0 else 0.5)
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
        self.sidebar.layout().addWidget(QtWidgets.QLabel("<b>Earth Curvature Profile (Bezier Graph)</b>"))
        self.bezier_editor = BezierCurveEditor()
        self.bezier_editor.curveChanged.connect(self.update_mesh_geometry)
        self.sidebar.layout().addWidget(self.bezier_editor)

        self.z_sld, self.z_spn = self.create_parameter_row("GLOBAL_Z", "Global Z Exaggeration (Pre)", 0.0, 10.0, params["GLOBAL_Z"])
        self.area_sld, self.area_spn = self.create_parameter_row("HORIZON_AREA_MAG", "Horizon Area Stretch (X/Y)", 0.0, 3.0, params["HORIZON_AREA_MAG"])
        self.relief_sld, self.relief_spn = self.create_parameter_row("HORIZON_RELIEF_MAG", "Horizon Relief Push (Z)", 1.0, 2.0, params["HORIZON_RELIEF_MAG"])

        self.level_btn = QtWidgets.QPushButton("LEVEL HORIZON VIEW")
        self.level_btn.setStyleSheet("background-color: #3182ce; color: #ffffff; font-weight: bold; padding: 6px; border-radius: 4px; margin-top: 5px;")
        self.level_btn.clicked.connect(self.level_camera_horizon)
        self.sidebar.layout().addWidget(self.level_btn)

        self.sidebar.layout().addWidget(QtWidgets.QLabel("\n" + "="*38 + "\nEXPORT TARGET CONFIGURATION"))
        self.w_file_txt = QtWidgets.QLineEdit(params["BLENDER_FILENAME"])
        self.sidebar.layout().addWidget(QtWidgets.QLabel("Warped Mesh Output Field Location:"))
        self.sidebar.layout().addWidget(self.w_file_txt)

        self.f_file_txt = QtWidgets.QLineEdit(params["FLAT_FILENAME"])
        self.sidebar.layout().addWidget(QtWidgets.QLabel("Flat Reference Mesh Output Field Location:"))
        self.sidebar.layout().addWidget(self.f_file_txt)
        
        self.bake_btn = QtWidgets.QPushButton("Bake Optimized Production OBJ")
        self.bake_btn.setStyleSheet("background-color: #1b4d3e; color: #ffffff; font-weight: bold; font-size: 11pt; padding: 10px; border-radius: 4px;")
        self.bake_btn.clicked.connect(self.execute_highres_production_bake)
        self.sidebar.layout().addWidget(self.bake_btn)

    def level_camera_horizon(self):
        focal = np.array(self.plotter_frame.camera.focal_point)
        pos = np.array(self.plotter_frame.camera.position)
        self.plotter_frame.camera.up = (0.0, 0.0, 1.0)
        self.plotter_frame.camera.focal_point = focal
        self.plotter_frame.camera.position = pos
        self.plotter_frame.render()

    def init_3d_canvas(self):
        X_w, Y_w, Z_w, r_orig, _ = run_berann_math(self.lons_preview, self.lats_preview, self.elevation_preview, params)
        
        self.preview_mesh = pv.StructuredGrid()
        self.preview_mesh.dimensions = (self.out_width, self.out_height, 1)
        self.preview_mesh.points = np.column_stack((X_w.ravel(), Y_w.ravel(), Z_w.ravel()))

        self.plotter_frame.set_background("dimgray")
        self.plotter_frame.add_mesh(self.preview_mesh, scalars=self.elevation_preview.ravel(), cmap="terrain", lighting=False, show_scalar_bar=False)

        row, col = np.unravel_index(np.argmin(r_orig), r_orig.shape)
        target_focal = np.array([X_w[row, col], Y_w[row, col], Z_w[row, col]])

        self.plotter_frame.camera.position = target_focal - np.array([0.0, CAMERA_DISTANCE, 0.0]) + np.array([0.0, 0.0, CAMERA_ALTITUDE])
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
        X_w, Y_w, Z_w, _, _ = run_berann_math(self.lons_preview, self.lats_preview, self.elevation_preview, params)
        self.preview_mesh.points = np.column_stack((X_w.ravel(), Y_w.ravel(), Z_w.ravel()))
        self.plotter_frame.render()

    def execute_highres_production_bake(self):
        params["BLENDER_FILENAME"] = self.w_file_txt.text()
        params["FLAT_FILENAME"] = self.f_file_txt.text()

        with rasterio.open(DEM_TIF_PATH) as prod_src:
            out_h = int(prod_src.height / PRODUCTION_DOWNSAMPLE_STEP)
            out_w = int(prod_src.width / PRODUCTION_DOWNSAMPLE_STEP)
            elevation_p = prod_src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            cols_p, rows_p = np.meshgrid(np.arange(0, prod_src.width, PRODUCTION_DOWNSAMPLE_STEP)[:out_w], np.arange(0, prod_src.height, PRODUCTION_DOWNSAMPLE_STEP)[:out_h])
            xs_p, ys_p = rasterio.transform.xy(prod_src.transform, rows_p, cols_p)
            lons_p, lats_p = np.array(xs_p).reshape(out_h, out_w), np.array(ys_p).reshape(out_h, out_w)

        X_p, Y_p, Z_p, r_p, dz_dr_p = run_berann_math(lons_p, lats_p, elevation_p, params)

        X_flat_raw = (lons_p - TARGET_LON) * (111320.0 * np.cos(np.radians(TARGET_LAT)))
        Y_flat_raw = (lats_p - TARGET_LAT) * 111320.0
        Z_flat_raw = elevation_p * params["GLOBAL_Z"]

        renderer = self.plotter_frame.renderer
        w_w, w_h = self.camera_cache["window_size"]

        homog_pts = np.hstack((np.column_stack((X_p.ravel(), Y_p.ravel(), Z_p.ravel())), np.ones((X_p.size, 1))))
        view_matrix = np.array(renderer.GetActiveCamera().GetModelViewTransformMatrix().GetData()).reshape(4,4)
        proj_matrix = np.array(renderer.GetActiveCamera().GetProjectionTransformMatrix(w_w / w_h, -1, 1).GetData()).reshape(4,4)

        clip_space = homog_pts @ (proj_matrix @ view_matrix).T
        w_component = clip_space[:, 3][:, None]
        w_component[w_component == 0] = 1e-5
        ndc = clip_space[:, :3] / w_component

        buf = VIEWPORT_BUFFER 
        in_view = ((ndc[:, 0] >= -(1.0 + buf)) & (ndc[:, 0] <= (1.0 + buf)) &  
                   (ndc[:, 1] >= -(1.0 + buf)) & (ndc[:, 1] <= (1.0 + buf)) &  
                   (ndc[:, 2] >= 0.0) & (ndc[:, 2] <= 1.0)).reshape(out_h, out_w)

        horizon_cutoff_height = params["BEZIER_P3"][1] * r_p.max()
        above_horizon = (Z_p >= horizon_cutoff_height)
        uncompressed = (np.abs(dz_dr_p) < 45.0)

        # Combine rules into a spatial structural mask
        master_valid_mask = in_view & above_horizon & uncompressed

        # Crop to the minimal bounding grid containing the valid items to maintain structural coordinates
        rows_valid, cols_valid = np.where(master_valid_mask)
        if len(rows_valid) == 0: return

        r_start, r_end = rows_valid.min(), rows_valid.max() + 1
        c_start, c_end = cols_valid.min(), cols_valid.max() + 1
        
        X_cropped, Y_cropped, Z_cropped = X_p[r_start:r_end, c_start:c_end], Y_p[r_start:r_end, c_start:c_end], Z_p[r_start:r_end, c_start:c_end]
        X_f_cropped, Y_f_cropped, Z_f_cropped = X_flat_raw[r_start:r_end, c_start:c_end], Y_flat_raw[r_start:r_end, c_start:c_end], Z_flat_raw[r_start:r_end, c_start:c_end]
        crop_mask = master_valid_mask[r_start:r_end, c_start:c_end]
        crop_height, crop_width = X_cropped.shape

        focal_world = self.camera_cache["focal_point"]
        SCALE_FACTOR = 10.0 / np.linalg.norm(focal_world - self.camera_cache["position"])

        pts_blender_warped = np.column_stack((
            (X_cropped.ravel() - focal_world[0]) * SCALE_FACTOR,
            (Z_cropped.ravel() - focal_world[2]) * SCALE_FACTOR,
            -((Y_cropped.ravel() - focal_world[1]) * SCALE_FACTOR)
        ))

        row_p, col_p = np.unravel_index(np.argmin(r_p), r_p.shape)
        pts_blender_flat = np.column_stack((
            X_f_cropped.ravel() * SCALE_FACTOR,
            (Z_f_cropped.ravel() - Z_flat_raw[row_p, col_p]) * SCALE_FACTOR,
            -(Y_f_cropped.ravel() * SCALE_FACTOR)
        ))

        u, v = np.linspace(0, 1, crop_width), np.linspace(1, 0, crop_height)
        uu, vv = np.meshgrid(u, v)
        uv_pts = np.column_stack((uu.ravel(), vv.ravel()))

        # Build faces based on structural mask evaluation to skip deleted points cleanly
        r_idx, c_idx = np.arange(crop_height - 1)[:, None], np.arange(crop_width - 1)
        v1 = r_idx * crop_width + c_idx + 1
        v2, v3 = v1 + 1, (r_idx + 1) * crop_width + c_idx + 2
        v4 = v3 - 1

        # Evaluate if all 4 corners of a quad face survived the threshold masks
        f1_valid = crop_mask[r_idx, c_idx]
        f2_valid = crop_mask[r_idx, c_idx + 1]
        f3_valid = crop_mask[r_idx + 1, c_idx + 1]
        f4_valid = crop_mask[r_idx + 1, c_idx]
        face_active_mask = (f1_valid & f2_valid & f3_valid & f4_valid).ravel()

        faces = np.column_stack((v1.ravel(), v4.ravel(), v3.ravel(), v2.ravel()))
        faces_filtered = faces[face_active_mask]
        faces_obj = np.column_stack((faces_filtered[:, 0], faces_filtered[:, 0], faces_filtered[:, 1], faces_filtered[:, 1], faces_filtered[:, 2], faces_filtered[:, 2], faces_filtered[:, 3], faces_filtered[:, 3]))

        for out_path, pts_data, tag in [(params["BLENDER_FILENAME"], pts_blender_warped, "Warped"), (params["FLAT_FILENAME"], pts_blender_flat, "Flat")]:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(f"# Clean Production Optimized {tag} Mesh\n")
                # Modified write format to receive explicit 2D sequences natively
                for pt in pts_data:
                    f.write(f"v {pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f}\n")
                np.savetxt(f, uv_pts, fmt="vt %.5f %.5f")
                np.savetxt(f, faces_obj, fmt="f %d/%d %d/%d %d/%d %d/%d")

    def closeEvent(self, event):
        self.plotter_frame.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())