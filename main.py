import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pyvistaqt import QtInteractor

from pipeline import CartographicPipeline
from widgets_map import MinimapWidget
from widgets_curves import DeformationControls

class ViewportProgressOverlay(QtWidgets.QWidget):
    """A classic, clean horizontal progress bar anchored to the bottom-center of the viewport."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 0, 40, 50) # Pads away from edges, anchors 50px from bottom
        layout.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter)
        
        # Classic sleek horizontal progress bar
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
        """Maintains overlay tracking relative to viewport dimensions."""
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
        
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)

        # Left Sidebar
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setFixedWidth(360)
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sidebar)

        # Main Plotter Container
        self.plotter_container = QtWidgets.QWidget()
        plotter_layout = QtWidgets.QGridLayout(self.plotter_container)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plotter = QtInteractor(self.plotter_container)
        self.plotter.set_background("#1e293b")
        if self.plotter.iren.get_interactor_style() is not None:
            self.plotter.iren.get_interactor_style().SetEnabled(False)
            
        plotter_layout.addWidget(self.plotter, 0, 0)
        layout.addWidget(self.plotter_container)

        # Centered inline overlay
        self.loading_overlay = ViewportProgressOverlay(self.plotter_container)

        self.create_menu_bar()
        self.build_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.loading_overlay.update_position()

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
        # Minimap
        map_group = QtWidgets.QGroupBox("Focal Alignment Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        self.minimap = MinimapWidget()
        self.minimap.centerChanged.connect(self.route_hardware_updates)
        map_layout.addWidget(self.minimap)
        self.sidebar_layout.addWidget(map_group)

        # Deformation Controls
        warp_group = QtWidgets.QGroupBox("Funnel Geometry Deformation Curve")
        warp_layout = QtWidgets.QVBoxLayout(warp_group)
        self.curves = DeformationControls()
        self.curves.valuesChanged.connect(self.route_hardware_updates)
        warp_layout.addWidget(self.curves)
        self.sidebar_layout.addWidget(warp_group)
        
        # Layer Selection Box
        layer_group = QtWidgets.QGroupBox("Active Rendering Layer Attribute")
        layer_layout = QtWidgets.QVBoxLayout(layer_group)
        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.addItems(["Hillshade", "Ambient Occlusion", "Texture Detail", "Vegetation Cover", "Landcover Class", "Soil Color Class"])
        self.layer_combo.currentIndexChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
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
            
            xmin, xmax, ymin, ymax, _, _ = self.pipeline.mesh.bounds
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

            self.pipeline._inject_shaders()
            self.pipeline.update_hardware_lut()
            self.route_hardware_updates()

        except Exception as rendering_err:
            QtWidgets.QMessageBox.critical(self, "GPU Mapping Error", f"Failed to instantiate render properties:\n\n{rendering_err}")

        self.import_action.setEnabled(True)
        self.export_action.setEnabled(True)
        self.loading_overlay.hide()

    def on_layer_changed(self, idx):
        if self._block_updates or not self.pipeline.mesh or not self.pipeline.mesh_actor: return
        self.pipeline.update_hardware_lut(idx)
        self.plotter.render()

    def route_hardware_updates(self):
        if self._block_updates or not self.pipeline.mesh or not self.pipeline.mesh_actor: return
        
        cx, cy = self.minimap.get_mesh_coords()
        amp = float(self.curves.amp_slider.value())
        decay = float(self.curves.decay_slider.value() / 10.0)
        alt = float(self.curves.alt_slider.value())

        xmin, xmax, ymin, ymax, _, _ = self.pipeline.mesh.bounds
        corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
        max_dist = float(np.max(np.sqrt((corners[:, 0] - cx)**2 + (corners[:, 1] - cy)**2)))
        if max_dist <= 0: max_dist = 1.0

        self.pipeline.update_shader_uniforms(cx, cy, max_dist, amp, decay)

        pts = self.pipeline.lightweight_points
        center_idx = np.argmin((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        z_base = pts[center_idx, 2]
        
        self.plotter.camera.position = (cx, cy, z_base + alt)
        self.plotter.camera.focal_point = (cx, cy, z_base)
        self.plotter.camera.up = (0.0, 1.0, 0.0)
        self.plotter.render()

    def on_export_masks(self):
        self.pipeline.execute_multipass_export(self.plotter, base_filename="berann_export")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())