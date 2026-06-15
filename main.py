import sys
import math
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pyvistaqt import QtInteractor

from pipeline import CartographicPipeline
from widgets_map import MinimapWidget
from widgets_curves import DeformationControls
from widgets_layer import GISLayerTreeWidget

class ViewportProgressOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 0, 40, 50)
        layout.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter)
        
        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setFixedSize(400, 20)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #1e293b;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 3px;
            }
        """)
        
        layout.addWidget(self.pbar)
        self.hide()

    def set_value(self, val):
        self.pbar.setValue(val)

    def update_position(self):
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())

class MeshLoadWorker(QtCore.QThread):
    progressChanged = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, pipeline, file_path, plotter):
        super().__init__()
        self.pipeline = pipeline
        self.file_path = file_path
        self.plotter = plotter

    def run(self):
        try:
            self.pipeline.load_mesh(
                self.file_path, 
                self.plotter, 
                progress_callback=self.progressChanged.emit
            )
            self.finished.emit(True, "Success")
        except Exception as e:
            self.finished.emit(False, str(e))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heinrich Berann Cartographic Projection Studio")
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

        self.pipeline = CartographicPipeline()
        self._block_updates = False
        self.worker = None
        self._settle_in_progress = False
        self._current_warp_profile = None
        self._update_settled_timer = QtCore.QTimer()
        self._update_settled_timer.setSingleShot(True)
        self._update_settled_timer.timeout.connect(self._on_active_updates_settled)

        self._camera_azimuth = 0.0
        self._camera_fov = 30.0
        self._drag_start = None
        
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)

        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setFixedWidth(360)
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sidebar)

        self.plotter_container = QtWidgets.QWidget()
        plotter_layout = QtWidgets.QGridLayout(self.plotter_container)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plotter = QtInteractor(self.plotter_container)
        self.plotter.set_background("#1e293b")
        if self.plotter.iren.get_interactor_style() is not None:
            self.plotter.iren.get_interactor_style().SetEnabled(False)
        self.plotter.installEventFilter(self)
            
        plotter_layout.addWidget(self.plotter, 0, 0)
        layout.addWidget(self.plotter_container)

        self.loading_overlay = ViewportProgressOverlay(self.plotter_container)

        self.create_menu_bar()
        self.build_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.loading_overlay.update_position()

    def eventFilter(self, obj, event):
        if obj == self.plotter:
            if event.type() == QtCore.QEvent.Wheel:
                delta = event.angleDelta().y()
                self._camera_fov = max(5.0, min(120.0, self._camera_fov - delta / 120 * 2))
                self.plotter.camera.SetViewAngle(self._camera_fov)
                self.plotter.render()
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
                        self.route_hardware_updates()
                    return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self._drag_start = None
                return True
        return super().eventFilter(obj, event)

    def _update_camera(self):
        if not self.pipeline.mesh:
            return
        cx, cy = self.minimap.get_mesh_coords()
        xmin, xmax, ymin, ymax, zmin, zmax = self.pipeline.mesh.bounds

        pts = self.pipeline.lightweight_points
        if pts is None or len(pts) == 0:
            z_base = (zmin + zmax) / 2.0
        else:
            center_idx = np.argmin((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            z_base = pts[center_idx, 2]

        height_factor = float(self.curves.alt_slider.value() / 100.0)
        if height_factor <= 0: height_factor = 0.1
        diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
        height = z_base + diagonal * height_factor

        el_rad = math.radians(float(self.curves.tilt_slider.value()))
        az_rad = math.radians(self._camera_azimuth)

        if el_rad > 0.01:
            cot_el = math.cos(el_rad) / math.sin(el_rad)
            look_dist_h = (height - z_base) * cot_el
            focal_x = cx + look_dist_h * math.sin(az_rad)
            focal_y = cy + look_dist_h * math.cos(az_rad)
            focal_z = z_base
        else:
            focal_x = cx + 10000 * math.sin(az_rad)
            focal_y = cy + 10000 * math.cos(az_rad)
            focal_z = z_base

        self.plotter.camera.position = (cx, cy, height)
        self.plotter.camera.focal_point = (focal_x, focal_y, focal_z)
        self.plotter.camera.up = (0.0, 0.0, 1.0)
        self.plotter.camera.clipping_range = (0.1, 100000.0)
        self.plotter.camera.SetViewAngle(self._camera_fov)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        self.import_action = QtWidgets.QAction("&Import Packed Face PLY...", self)
        self.import_action.setShortcut("Ctrl+I")
        self.import_action.triggered.connect(self.on_load_mesh)
        
        self.export_action = QtWidgets.QAction("&Export Multi-Pass Masks...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.setEnabled(False)  
        self.export_action.triggered.connect(self.on_export_masks)

        exit_action = QtWidgets.QAction("&Exit Studio Window", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(self.import_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def build_ui(self):
        map_group = QtWidgets.QGroupBox("Focal Alignment Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        self.minimap = MinimapWidget()
        self.minimap.centerChanged.connect(self.route_hardware_updates)
        map_layout.addWidget(self.minimap)
        self.sidebar_layout.addWidget(map_group)

        warp_group = QtWidgets.QGroupBox("Deformation & View Controls")
        warp_layout = QtWidgets.QVBoxLayout(warp_group)
        self.curves = DeformationControls()
        self.curves.warpProfileChanged.connect(self._on_warp_profile)
        self.curves.alt_slider.valueChanged.connect(self.route_hardware_updates)
        self.curves.tilt_slider.valueChanged.connect(self.route_hardware_updates)
        warp_layout.addWidget(self.curves)
        self.sidebar_layout.addWidget(warp_group)
        
        layer_group = QtWidgets.QGroupBox("Active Rendering Layer Attribute")
        layer_layout = QtWidgets.QVBoxLayout(layer_group)
        self.layer_tree = GISLayerTreeWidget(self.pipeline, self.plotter)
        self.layer_tree.layerChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_tree)
        self.sidebar_layout.addWidget(layer_group)
        
        self.sidebar_layout.addStretch()

    def on_load_mesh(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Target Geometry PLY", "", "PLY Meshes (*.ply)")
        if not file_path: return
        
        self.import_action.setEnabled(False)
        self.export_action.setEnabled(False)
        
        self.loading_overlay.set_value(0)
        self.loading_overlay.update_position()
        self.loading_overlay.show()

        self.worker = MeshLoadWorker(self.pipeline, file_path, self.plotter)
        self.worker.progressChanged.connect(self.handle_worker_progress)
        self.worker.finished.connect(self.handle_worker_finished)
        self.worker.start()

    def handle_worker_progress(self, percent, message):
        self.loading_overlay.set_value(percent)

    def handle_worker_finished(self, success, error_message):
        self.loading_overlay.set_value(100)
        
        if not success:
            QtWidgets.QMessageBox.critical(self, "Pipeline Load Failure", f"An error occurred during disk streaming:\n\n{error_message}")
            self.import_action.setEnabled(True)
            self.loading_overlay.hide()
            return

        QtWidgets.QApplication.processEvents()

        try:
            self.minimap.compute_dem_from_downsampled(self.pipeline.lightweight_points)
            
            xmin, xmax, ymin, ymax, zmin, zmax = self.pipeline.mesh.bounds

            diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            
            self._block_updates = True
            self.curves.calibrate_ranges(diagonal)
            self._block_updates = False

            self.pipeline.mesh_actor = self.plotter.add_mesh(
                self.pipeline.mesh,
                show_scalar_bar=False,
                rgb=False,
                scalars="Hillshade"
            )

            prop = self.pipeline.mesh_actor.GetProperty()
            prop.SetLighting(True)
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
            prop.BackfaceCullingOff()
            prop.FrontfaceCullingOff()
            prop.SetInterpolationToGouraud()

            self.pipeline.update_hardware_lut()

            # Create proxy mesh in main thread (VTK threading safety)
            self.loading_overlay.set_value(95)
            self.loading_overlay.show()
            QtWidgets.QApplication.processEvents()
            self.pipeline._create_proxy_mesh(divisions=200)
            QtWidgets.QApplication.processEvents()

            # Pre-seed warp buffers for both proxy and full mesh
            seed_profile = np.zeros(256, dtype=np.float32)
            self.pipeline._warp_proxy_cpu(0.0, 0.0, 0.0, seed_profile)
            self.pipeline._warp_mesh_cpu(0.0, 0.0, 0.0, seed_profile)

            self.plotter.reset_camera()
            
            self.route_hardware_updates()

        except Exception as rendering_err:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "GPU Mapping Error", f"Failed to instantiate render properties:\n\n{rendering_err}")

        self.import_action.setEnabled(True)
        self.export_action.setEnabled(True)
        self.loading_overlay.hide()

    def on_layer_changed(self, idx):
        if self._block_updates or not self.pipeline.mesh or not self.pipeline.mesh_actor: return
        self.pipeline.update_hardware_lut(idx)
        self.plotter.render()

    def _on_warp_profile(self, profile):
        self._current_warp_profile = profile
        self.route_hardware_updates()

    def route_hardware_updates(self, *args):
        if self._block_updates or not self.pipeline.mesh or not self.pipeline.mesh_actor:
            return

        import traceback
        try:
            cx, cy = self.minimap.get_mesh_coords()
            view_angle = self._camera_azimuth
            profile = self._current_warp_profile
            if profile is None:
                return

            self.pipeline._warp_proxy_cpu(cx, cy, view_angle, profile)
            if not self.pipeline._using_proxy:
                self.pipeline.swap_to_proxy()

            self.pipeline.mesh_actor.GetShaderProperty().ClearAllShaderReplacements()

            self._update_camera()
            self.minimap.set_view_angle(self._camera_azimuth)

            self.pipeline.mesh_actor.Modified()
            self.plotter.render()
            self._update_settled_timer.start(400)

        except Exception as e:
            traceback.print_exc()

    def _on_active_updates_settled(self):
        if self._settle_in_progress or not self.pipeline.mesh or not self.pipeline.mesh_actor:
            return
        self._settle_in_progress = True
        try:
            if self._block_updates:
                return

            cx, cy = self.minimap.get_mesh_coords()
            view_angle = self._camera_azimuth
            profile = self._current_warp_profile
            if profile is None:
                return

            self.pipeline._warp_mesh_cpu(cx, cy, view_angle, profile)
            self.pipeline.swap_to_full()
            self.plotter.render()

        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            self._settle_in_progress = False

    def on_export_masks(self):
        self.pipeline.execute_multipass_export(self.plotter, base_filename="berann_export")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
