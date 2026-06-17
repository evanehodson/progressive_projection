import sys
import math
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from camera import BerannCamera, CameraState
from raster_stack import RasterStack
from raymarch import raymarch
from compositor import compose_viewport, gbuffer_to_qimage
from widgets_map import MinimapWidget
from widgets_curves import DeformationControls
from widgets_layer import GISLayerTreeWidget

DEM_PATH = "../data/wurl/processed_dem.tif"
SHADOW_PATH = "../data/wurl/shadow_ao.tif"
LC_PATH = "../data/wurl/unified_landcover.tif"
SOIL_PATH = "../data/wurl/aligned_soil_raster.tif"


class ViewportWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._image = None

    def set_image(self, rgba):
        self._image = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                                   QtGui.QImage.Format_RGBA8888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self._image is not None and not self._image.isNull():
            painter.drawImage(self.rect(), self._image)
        else:
            painter.fillRect(self.rect(), QtGui.QColor("#1e293b"))
            painter.setPen(QtGui.QColor("#475569"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No render")

    def resizeEvent(self, event):
        super().resizeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Berann 2.5D Studio")
        self.resize(1500, 950)
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; color: #f8fafc; }
            QMenuBar { background-color: #1e293b; color: #f8fafc; font-weight: bold; border-bottom: 1px solid #334155; }
            QMenuBar::item:selected { background-color: #2563eb; }
            QMenu { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; }
            QMenu::item:selected { background-color: #2563eb; }
            QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 6px; margin-top: 8px; padding: 6px; color: #f8fafc; }
            QLabel { color: #cbd5e1; font-size: 8pt; }
            QComboBox { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px; }
        """)

        self._block_updates = False
        self._camera_azimuth = 0.0
        self._camera_fov = 30.0
        self._drag_start = None
        self._current_warp_profile = None
        self._settle_in_progress = False
        self._gbuffer_cache = None

        self._update_settled_timer = QtCore.QTimer()
        self._update_settled_timer.setSingleShot(True)
        self._update_settled_timer.timeout.connect(self._on_settled)

        # --- RASTER STACK ---
        print("Loading raster stack...", flush=True)
        self.stack = RasterStack(DEM_PATH, SHADOW_PATH, LC_PATH, SOIL_PATH)

        # --- CAMERA ---
        self.camera = BerannCamera(
            self.stack.x_min, self.stack.x_max,
            self.stack.y_min, self.stack.y_max
        )

        z_base = self.stack.get_elevation_at(
            (self.stack.x_min + self.stack.x_max) / 2.0,
            (self.stack.y_min + self.stack.y_max) / 2.0
        )
        self.camera.update(CameraState(), z_base=z_base)

        # --- UI ---
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)

        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setFixedWidth(360)
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sidebar)

        # Viewport
        self.viewport = ViewportWidget()
        self.viewport.installEventFilter(self)
        layout.addWidget(self.viewport, 1)

        self.create_menu_bar()
        self.build_ui()

        # Initial render
        self._render_viewport(quality="settled")

    # ---- Event handling ----

    def eventFilter(self, obj, event):
        if obj == self.viewport:
            if event.type() == QtCore.QEvent.Wheel:
                delta = event.angleDelta().y()
                self._camera_fov = max(5.0, min(120.0, self._camera_fov - delta / 120 * 2))
                self.route_updates(interactive=True)
                return True
            elif event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self._drag_start = event.pos()
                self._drag_azimuth = self._camera_azimuth
                return True
            elif event.type() == QtCore.QEvent.MouseMove and self._drag_start is not None:
                if event.buttons() & QtCore.Qt.LeftButton:
                    dx = event.pos().x() - self._drag_start.x()
                    sensitivity = 0.3
                    new_az = (self._drag_azimuth - dx * sensitivity) % 360
                    if abs(new_az - self._camera_azimuth) > 0.01:
                        self._camera_azimuth = new_az
                        self.minimap.set_view_angle(self._camera_azimuth)
                        self.route_updates(interactive=True)
                    return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self._drag_start = None
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    # ---- UI ----

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        export_action = QtWidgets.QAction("&Export G-Buffer...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)
        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def build_ui(self):
        map_group = QtWidgets.QGroupBox("Focal Alignment Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        self.minimap = MinimapWidget()
        self.minimap.centerChanged.connect(lambda: self.route_updates(interactive=True))
        map_layout.addWidget(self.minimap)
        self.sidebar_layout.addWidget(map_group)

        # Feed DEM to minimap
        self._init_minimap()

        warp_group = QtWidgets.QGroupBox("Deformation & View Controls")
        warp_layout = QtWidgets.QVBoxLayout(warp_group)
        self.curves = DeformationControls()
        self.curves.warpProfileChanged.connect(self._on_warp_profile)
        self.curves.alt_slider.valueChanged.connect(lambda: self.route_updates(interactive=True))
        self.curves.tilt_slider.valueChanged.connect(lambda: self.route_updates(interactive=True))
        warp_layout.addWidget(self.curves)
        self.sidebar_layout.addWidget(warp_group)

        # Calibrate ranges
        self.curves.calibrate_ranges(self.camera.diagonal)
        self._current_warp_profile = self.curves.curve_editor._sample_curve() * (self.camera.diagonal * 0.25)

        layer_group = QtWidgets.QGroupBox("Active Rendering Layer")
        layer_layout = QtWidgets.QVBoxLayout(layer_group)
        self.layer_tree = GISLayerTreeWidget(None, None)
        self.layer_tree.layerChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self.layer_tree)
        self.sidebar_layout.addWidget(layer_group)
        self.sidebar_layout.addStretch()

    def _init_minimap(self):
        dem = self.stack.dem_filled
        H, W = dem.shape
        stride = max(1, H // 256, W // 256)
        rows = np.arange(0, H, stride)
        cols_arr = np.arange(0, W, stride)
        g_rows, g_cols = np.meshgrid(rows, cols_arr, indexing='ij')
        n = g_rows.size
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = self.stack.x_min + g_cols.ravel() * self.stack.res
        pts[:, 1] = self.stack.y_max - g_rows.ravel() * self.stack.res
        pts[:, 2] = dem[g_rows, g_cols].ravel()
        self.minimap.compute_dem_from_downsampled(pts)

    # ---- Rendering ----

    def _render_viewport(self, quality="settled"):
        vp_w = self.viewport.width()
        vp_h = self.viewport.height()
        if vp_w < 10 or vp_h < 10:
            return

        if quality == "interactive":
            out_w = vp_w // 2
            out_h = vp_h // 2
            tier = "interactive"
        else:
            out_w = vp_w
            out_h = vp_h
            tier = "settled"

        if self._current_warp_profile is None:
            return

        state = CameraState(
            cx=self.minimap.get_mesh_coords()[0],
            cy=self.minimap.get_mesh_coords()[1],
            azimuth=self._camera_azimuth,
            tilt=self.curves.tilt_slider.value(),
            height_factor=self.curves.alt_slider.value() / 100.0,
            fov=self._camera_fov,
            profile=self._current_warp_profile,
        )

        z_base = self.stack.get_elevation_at(state.cx, state.cy)
        self.camera.update(state, z_base=z_base)

        gbuffer = raymarch(self.stack, self.camera, out_w, out_h, quality_tier=tier)
        self._gbuffer_cache = gbuffer

        rgba = compose_viewport(gbuffer)
        self.viewport.set_image(rgba)

    # ---- Callbacks ----

    def _on_warp_profile(self, profile):
        self._current_warp_profile = profile
        self.route_updates(interactive=True)

    def route_updates(self, interactive=False):
        if self._block_updates:
            return
        if interactive:
            self._render_viewport(quality="interactive")
        else:
            self._render_viewport(quality="settled")
        self._update_settled_timer.start(400)

    def _on_settled(self):
        if self._settle_in_progress:
            return
        self._settle_in_progress = True
        try:
            self._render_viewport(quality="settled")
        finally:
            self._settle_in_progress = False

    def _on_layer_changed(self, idx):
        pass

    def _on_export(self):
        if self._current_warp_profile is None or self._gbuffer_cache is None:
            return

        export_w = 16384
        export_h = 8192

        state = CameraState(
            cx=self.minimap.get_mesh_coords()[0],
            cy=self.minimap.get_mesh_coords()[1],
            azimuth=self._camera_azimuth,
            tilt=self.curves.tilt_slider.value(),
            height_factor=self.curves.alt_slider.value() / 100.0,
            fov=self._camera_fov,
            profile=self._current_warp_profile,
        )
        z_base = self.stack.get_elevation_at(state.cx, state.cy)
        self.camera.update(state, z_base=z_base)

        print(f"Exporting G-buffer at {export_w}x{export_h}...", flush=True)
        gbuffer = raymarch(self.stack, self.camera, export_w, export_h, quality_tier="export")
        rgba = compose_viewport(gbuffer)

        from PIL import Image
        img = Image.frombuffer("RGBA", (export_w, export_h), rgba, "raw", "RGBA", 0, 1)
        img.save("berann_export_viewport.png")
        print("Exported: berann_export_viewport.png", flush=True)
        self._gbuffer_cache = gbuffer


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
