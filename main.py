import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pyvistaqt import QtInteractor

from pipeline import CartographicPipeline
from widgets_map import MinimapWidget
from widgets_curves import DeformationControls

class MeshLoadWorker(QtCore.QThread):
    """Worker thread to handle the 1.18 GB disk load without freezing the PyQt GUI."""
    progressChanged = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, pipeline, file_path, plotter):
        super().__init__()
        self.pipeline = pipeline
        self.file_path = file_path
        self.plotter = plotter

    def run(self):
        try:
            # We pass our thread's signal callback into the pipeline
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
            QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 6px; margin-top: 8px; padding: 6px; color: #f8fafc; }
            QLabel { color: #cbd5e1; font-size: 8pt; }
            QPushButton { background-color: #2563eb; color: white; border: none; padding: 6px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #3b82f6; }
            QPushButton:disabled { background-color: #1e293b; color: #64748b; border: 1px solid #334155; }
            QComboBox { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px; }
            QProgressBar { border: 1px solid #334155; border-radius: 4px; text-align: center; background-color: #1e293b; color: white; font-weight: bold; }
            QProgressBar::chunk { background-color: #10b981; width: 10px; }
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
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sidebar)

        # Main Plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e293b")
        if self.plotter.iren.get_interactor_style() is not None:
            self.plotter.iren.get_interactor_style().SetEnabled(False) # Lock standard movement
        layout.addWidget(self.plotter)

        self.build_ui()

    def build_ui(self):
        # IO Block
        io_group = QtWidgets.QGroupBox("Mesh Entry")
        io_layout = QtWidgets.QVBoxLayout(io_group)
        self.load_btn = QtWidgets.QPushButton("Load Packed Face PLY...")
        self.load_btn.clicked.connect(self.on_load_mesh)
        self.export_btn = QtWidgets.QPushButton("Execute Multi-Pass Export")
        self.export_btn.clicked.connect(self.on_export_masks)
        
        # New Progress Monitoring UI Elements
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.status_lbl = QtWidgets.QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #94a3b8; font-style: italic;")

        io_layout.addWidget(self.load_btn)
        io_layout.addWidget(self.export_btn)
        io_layout.addWidget(self.progress_bar)
        io_layout.addWidget(self.status_lbl)
        self.sidebar.layout().addWidget(io_group)

        # Minimap
        map_group = QtWidgets.QGroupBox("Focal Alignment Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        self.minimap = MinimapWidget()
        self.minimap.centerChanged.connect(self.route_hardware_updates)
        map_layout.addWidget(self.minimap)
        self.sidebar.layout().addWidget(map_group)

        # Deformation Controls
        warp_group = QtWidgets.QGroupBox("Funnel Geometry Deformation Curve")
        warp_layout = QtWidgets.QVBoxLayout(warp_group)
        self.curves = DeformationControls()
        self.curves.valuesChanged.connect(self.route_hardware_updates)
        warp_layout.addWidget(self.curves)
        self.sidebar.layout().addWidget(warp_group)
        
        # Layer Selection Box
        layer_group = QtWidgets.QGroupBox("Active Rendering Layer Attribute")
        layer_layout = QtWidgets.QVBoxLayout(layer_group)
        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.addItems(["Hillshade", "Ambient Occlusion", "Texture Detail", "Vegetation Cover", "Landcover Class", "Soil Color Class"])
        self.layer_combo.currentIndexChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        self.sidebar.layout().addWidget(layer_group)

    def on_load_mesh(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Target Geometry PLY", "", "PLY Meshes (*.ply)")
        if not file_path: return
        
        # Lock buttons and display loading UI
        self.load_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_lbl.setText("Initializing memory streams...")

        # Fire off the worker thread
        self.worker = MeshLoadWorker(self.pipeline, file_path, self.plotter)
        self.worker.progressChanged.connect(self.handle_worker_progress)
        self.worker.finished.connect(self.handle_worker_finished)
        self.worker.start()

    def handle_worker_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_lbl.setText(message)

    def handle_worker_finished(self, success, error_message):
        self.progress_bar.setValue(100)
        
        if not success:
            self.status_lbl.setText("[CRASH PREVENTED]")
            QtWidgets.QMessageBox.critical(self, "Pipeline Load Failure", f"An error occurred during disk streaming:\n\n{error_message}")
            self.load_btn.setEnabled(True)
            self.progress_bar.hide()
            return

        self.status_lbl.setText("Binding actors to VTK Graphic Pipeline...")
        QtWidgets.QApplication.processEvents()

        # Finalizing: Binding actor contexts to the viewport must happen here 
        # inside the safe primary GUI thread
        try:
            self.minimap.compute_dem_from_downsampled(self.pipeline.lightweight_points)
            
            xmin, xmax, ymin, ymax, _, _ = self.pipeline.mesh.bounds
            diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            
            self._block_updates = True
            self.curves.calibrate_ranges(diagonal)
            self._block_updates = False

            # Bind actual mesh object inside renderer targets
            self.pipeline.mesh_actor = self.plotter.add_mesh(
                self.pipeline.mesh,
                show_scalar_bar=False,
                rgb=False,
                scalars="Hillshade"
            )

            # Re-apply static material shading parameters
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
            
            self.status_lbl.setText("Ready (Ingestion Completed)")

        except Exception as rendering_err:
            QtWidgets.QMessageBox.critical(self, "GPU Mapping Error", f"Failed to instantiate render properties:\n\n{rendering_err}")
            self.status_lbl.setText("Pipeline Error")

        # Unlock application controls
        self.load_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.hide()

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