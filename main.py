import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from pyvistaqt import QtInteractor

from pipeline import CartographicPipeline
from widgets_map import MinimapWidget
from widgets_curves import DeformationControls
from widgets_layer import GISLayerTreeWidget

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
        print(f"[MAIN WINDOW] Loader finished thread execution. Success Flag={success}")
        sys.stdout.flush()
        
        if not success:
            QtWidgets.QMessageBox.critical(self, "Pipeline Load Failure", f"An error occurred during disk streaming:\n\n{error_message}")
            self.import_action.setEnabled(True)
            self.loading_overlay.hide()
            return

        QtWidgets.QApplication.processEvents()

        try:
            print(f"[MAIN WINDOW] Computing minimap DEM context arrays...")
            self.minimap.compute_dem_from_downsampled(self.pipeline.lightweight_points)
            
            xmin, xmax, ymin, ymax, zmin, zmax = self.pipeline.mesh.bounds
            print(f"[MAIN WINDOW] Loaded Data Spatial Constraints:")
            print(f"    -> X Bounds: {xmin} to {xmax}")
            print(f"    -> Y Bounds: {ymin} to {ymax}")
            print(f"    -> Z Bounds: {zmin} to {zmax}")
            sys.stdout.flush()

            diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            
            self._block_updates = True
            self.curves.calibrate_ranges(diagonal)
            self._block_updates = False

            print("[MAIN WINDOW] Handing dataset reference off to pyvista.Plotter.add_mesh()...")
            self.pipeline.mesh_actor = self.plotter.add_mesh(
                self.pipeline.mesh,
                show_scalar_bar=False,
                rgb=False,
                scalars="Hillshade"
            )
            
            if self.pipeline.mesh_actor is None:
                print("[FATAL DIAGNOSTIC] pyvista.Plotter.add_mesh returned a completely NULL actor object reference!")
            else:
                print(f"[MAIN WINDOW] Actor wrapped successfully. Class={self.pipeline.mesh_actor.GetClassName()}")
            sys.stdout.flush()

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
            
            print("[MAIN WINDOW] Requesting plotter to automatically reset view matrix onto actor bounds...")
            self.plotter.reset_camera()
            print(f"[MAIN WINDOW] Camera state post-reset: Pos={self.plotter.camera.position}, Focus={self.plotter.camera.focal_point}")
            
            self.route_hardware_updates()

        except Exception as rendering_err:
            print(f"[CRITICAL ERROR EXCEPTION] Failure during pipeline canvas layout build: {rendering_err}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            QtWidgets.QMessageBox.critical(self, "GPU Mapping Error", f"Failed to instantiate render properties:\n\n{rendering_err}")

        self.import_action.setEnabled(True)
        self.export_action.setEnabled(True)
        self.loading_overlay.hide()

    def on_layer_changed(self, idx):
        if self._block_updates or not self.pipeline.mesh or not self.pipeline.mesh_actor: return
        self.pipeline.update_hardware_lut(idx)
        self.plotter.render()

    def route_hardware_updates(self, *args):
        if self._block_updates or not self.pipeline.mesh or not self.pipeline.mesh_actor: 
            print("[DEBUG] Guard triggered: updates are blocked or mesh elements are missing.")
            return
        
        print("\n=== [MINIMAP INTERACTION TRACE] ===")
        try:
            # 1. Fetch current user controls
            cx, cy = self.minimap.get_mesh_coords()
            amp = float(self.curves.amp_slider.value())
            decay = float(self.curves.decay_slider.value() / 10.0)
            
            alt_factor = float(self.curves.alt_slider.value() / 100.0)
            if alt_factor <= 0: alt_factor = 0.1

            print(f" -> Normalized inputs clicked: cx={self.minimap.cx:.4f}, cy={self.minimap.cy:.4f}")
            print(f" -> Extrapolated mesh target:  X={cx:.2f}, Y={cy:.2f}")

            # 2. Extract bounding radius constraints
            xmin, xmax, ymin, ymax, zmin, zmax = self.pipeline.mesh.bounds
            print(f" -> Real VTK Mesh Bounds:      X:[{xmin:.1f} to {xmax:.1f}], Y:[{ymin:.1f} to {ymax:.1f}], Z:[{zmin:.1f} to {zmax:.1f}]")

            # Check if coordinates mapped outside real spatial limits
            if not (xmin <= cx <= xmax) or not (ymin <= cy <= ymax):
                print(" [⚠️ WARNING] Target coordinate is completely OUTSIDE actual mesh boundaries!")

            corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
            max_dist = float(np.max(np.sqrt((corners[:, 0] - cx)**2 + (corners[:, 1] - cy)**2)))
            if max_dist <= 0: max_dist = 1.0

            # 3. Stream uniform parameters straight to the hardware actor shader uniforms
            print(f" -> Sending Shader Uniforms: Center=({cx:.2f}, {cy:.2f}), MaxDist={max_dist:.2f}, Amp={amp:.2f}, Decay={decay:.2f}")
            self.pipeline.update_shader_uniforms(cx, cy, max_dist, amp, decay)

            # 4. Handle proportional view matrix tracking
            pts = self.pipeline.lightweight_points
            if pts is None or len(pts) == 0:
                print(" [⚠️ WARNING] pipeline.lightweight_points is empty! Falling back to flat bounding center.")
                z_base = (zmin + zmax) / 2.0
            else:
                center_idx = np.argmin((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
                z_base = pts[center_idx, 2]
                print(f" -> Nearest node tracking:      Index={center_idx}, base Z elevation={z_base:.2f}")
            
            diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            camera_distance = diagonal * alt_factor

            pos_x = cx
            pos_y = cy - (camera_distance * 0.4)
            pos_z = z_base + camera_distance

            print(f" -> Calculated Distance Scaling: Diagonal={diagonal:.2f}, CamDistance={camera_distance:.2f}")
            print(f" -> Target Camera Position:      ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})")
            print(f" -> Target Camera Focal Point:   ({cx:.2f}, {cy:.2f}, {z_base:.2f})")

            # Check for non-finite values that shatter the projection matrix
            if not np.all(np.isfinite([pos_x, pos_y, pos_z, cx, cy, z_base])):
                print(" [❌ ERROR] Non-finite float value (NaN or Inf) detected in camera layout coordinates!")

            self.plotter.camera.position = (pos_x, pos_y, pos_z)
            self.plotter.camera.focal_point = (cx, cy, z_base)
            self.plotter.camera.up = (0.0, 0.0, 1.0)
            
            # Keep manual expansive depth limits for custom vertex shader displacements
            self.plotter.camera.clipping_range = (0.1, 1000.0)
            self.plotter.camera.Modified()
            
            # === DEEP STATE VERIFICATION DEBUG LOGS ===
            print(f" [PIPELINE TRACE] Actor Visibility: {self.pipeline.mesh_actor.GetVisibility()}")
            print(f" [PIPELINE TRACE] Actor Mapper Address: {self.pipeline.mesh_actor.GetMapper()}")
            print(f" [PIPELINE TRACE] Camera Manual Clipping Range: {self.plotter.camera.clipping_range}")
            print(f" [PIPELINE TRACE] Camera View Angle: {self.plotter.camera.view_angle}")
            
            # FIX: Bypass PyVista wrapper object and extract 4x4 matrix elements safely 
            # straight from the C++ underlying vtkCamera pointer.
            vtk_cam = self.plotter.camera
            m = vtk_cam.GetModelViewTransformMatrix()
            print(" [PIPELINE TRACE] Native View Matrix Structural Orientation Layout:")
            print(f"    [{m.GetElement(0,0):.4f}, {m.GetElement(0,1):.4f}, {m.GetElement(0,2):.4f}, {m.GetElement(0,3):.4f}]")
            print(f"    [{m.GetElement(1,0):.4f}, {m.GetElement(1,1):.4f}, {m.GetElement(1,2):.4f}, {m.GetElement(1,3):.4f}]")
            print(f"    [{m.GetElement(2,0):.4f}, {m.GetElement(2,1):.4f}, {m.GetElement(2,2):.4f}, {m.GetElement(2,3):.4f}]")
            print(f"    [{m.GetElement(3,0):.4f}, {m.GetElement(3,1):.4f}, {m.GetElement(3,2):.4f}, {m.GetElement(3,3):.4f}]")
            # ==========================================

            # 5. Redraw viewport canvas
            self.plotter.render()
            print(" -> Status: Render pipeline execution passed successfully.")

        except Exception as e:
            import traceback
            print(" [❌ CRITICAL EXCEPTION ENCOUNTERED]")
            traceback.print_exc()
            
        print("====================================\n")

    def on_export_masks(self):
        self.pipeline.execute_multipass_export(self.plotter, base_filename="berann_export")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())