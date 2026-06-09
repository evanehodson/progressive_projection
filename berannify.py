import sys
import numpy as np
import pyvista as pv
from plyfile import PlyData
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore, QtGui


class MinimapWidget(QtWidgets.QWidget):
    """
    Low-resolution, anti-aliased grayscale DEM view.
    """
    centerChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setMaximumSize(300, 300)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.cx = 0.5  
        self.cy = 0.5  
        self.dem_image = None
        self.xmin, self.xmax = 0.0, 1.0
        self.ymin, self.ymax = 0.0, 1.0

    def compute_dem_from_downsampled(self, pts_downsampled):
        if pts_downsampled is None or len(pts_downsampled) == 0:
            self.dem_image = None
            self.update()
            return

        self.xmin, self.xmax = pts_downsampled[:, 0].min(), pts_downsampled[:, 0].max()
        self.ymin, self.ymax = pts_downsampled[:, 1].min(), pts_downsampled[:, 1].max()

        res = 128 
        grid_z = np.zeros((res, res), dtype=np.float32)
        grid_counts = np.zeros((res, res), dtype=np.float32)

        x_indices = ((pts_downsampled[:, 0] - self.xmin) / (self.xmax - self.xmin) * (res - 1)).astype(np.int32)
        y_indices = ((pts_downsampled[:, 1] - self.ymin) / (self.ymax - self.ymin) * (res - 1)).astype(np.int32)

        x_indices = np.clip(x_indices, 0, res - 1)
        y_indices = np.clip(y_indices, 0, res - 1)

        for i in range(len(pts_downsampled)):
            grid_z[y_indices[i], x_indices[i]] += pts_downsampled[i, 2]
            grid_counts[y_indices[i], x_indices[i]] += 1.0

        valid_mask = grid_counts > 0
        grid_z[valid_mask] /= grid_counts[valid_mask]
        
        if not np.all(valid_mask):
            mean_val = grid_z[valid_mask].mean() if np.any(valid_mask) else 0.0
            grid_z[~valid_mask] = mean_val

        z_min, z_max = grid_z.min(), grid_z.max()
        if z_max != z_min:
            normalized_z = ((grid_z - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)
        else:
            normalized_z = np.zeros((res, res), dtype=np.uint8)

        self.dem_image = QtGui.QImage(res, res, QtGui.QImage.Format_Grayscale8)
        flipped_z = np.flipud(normalized_z)
        for y in range(res):
            for x in range(res):
                self.dem_image.setPixel(x, y, int(flipped_z[y, x]))
        self.update()

    def get_mesh_coords(self):
        mx = self.xmin + self.cx * (self.xmax - self.xmin)
        my = self.ymin + self.cy * (self.ymax - self.ymin)
        return mx, my

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#1e293b"))
        pad = 4
        w, h = self.width() - 2*pad, self.height() - 2*pad
        if self.dem_image is not None:
            target = QtCore.QRect(pad, pad, w, h)
            painter.drawImage(target, self.dem_image)
        hx = int(pad + self.cx * w)
        hy = int(pad + (1.0 - self.cy) * h)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ef4444"), 1, QtCore.Qt.DashLine))
        painter.drawLine(0, hy, self.width(), hy)
        painter.drawLine(hx, 0, hx, self.height())
        painter.setBrush(QtGui.QColor("#ef4444"))
        painter.drawEllipse(QtCore.QPoint(hx, hy), 4, 4)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton: self.update_from_mouse(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton: self.update_from_mouse(event.pos())

    def update_from_mouse(self, pos):
        pad = 4
        w, h = self.width() - 2*pad, self.height() - 2*pad
        if w <= 0 or h <= 0: return
        self.cx = np.clip((pos.x() - pad) / w, 0.0, 1.0)
        self.cy = np.clip(1.0 - ((pos.y() - pad) / h), 0.0, 1.0)
        self.update()
        self.centerChanged.emit(self.cx, self.cy)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Heinrich Berann Cartographic Projection Studio")
        self.resize(1500, 950)

        self.mesh = None
        self.mesh_actor = None
        self._block_updates = False
        self.lightweight_points = None
        
        # Default palettes for discrete classes
        self.lc_colors = {0: "#1e3a8a", 1: "#22c55e", 2: "#eab308", 3: "#ef4444", 4: "#71717a"}
        self.soil_colors = {0: "#78350f", 1: "#b45309", 2: "#d97706", 3: "#a1a1aa"}

        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; color: #f8fafc; }
            QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 6px; margin-top: 8px; padding: 6px; color: #f8fafc; }
            QLabel { color: #cbd5e1; font-size: 8pt; }
            QPushButton { background-color: #2563eb; color: white; border: none; padding: 6px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #3b82f6; }
            QListWidget { background-color: #1e293b; border: 1px solid #334155; border-radius: 4px; color: #f8fafc; }
        """)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QHBoxLayout(main_widget)

        # Left Sidebar (Controls)
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setFixedWidth(360)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.sidebar)

        # Main Plotter
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e293b")
        if self.plotter.iren.get_interactor_style() is not None:
            self.plotter.iren.get_interactor_style().SetEnabled(False)
        layout.addWidget(self.plotter)

        self.build_ui_modules()

    def build_ui_modules(self):
        # IO Block
        io_group = QtWidgets.QGroupBox("Mesh Entry")
        io_layout = QtWidgets.QVBoxLayout(io_group)
        self.load_btn = QtWidgets.QPushButton("Load Packed Face PLY...")
        self.load_btn.clicked.connect(self.on_load_mesh_clicked)
        io_layout.addWidget(self.load_btn)
        self.sidebar.layout().addWidget(io_group)

        # Minimap Block
        map_group = QtWidgets.QGroupBox("Focal Alignment Map")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        self.minimap = MinimapWidget()
        self.minimap.centerChanged.connect(self.on_minimap_center_updated)
        map_layout.addWidget(self.minimap)
        self.sidebar.layout().addWidget(map_group)

        # Layer Selection Tree (Exclusive Mutex Checkboxes)
        layer_group = QtWidgets.QGroupBox("Active Rendering Layer Attribute")
        layer_layout = QtWidgets.QVBoxLayout(layer_group)
        self.layer_list = QtWidgets.QListWidget()
        
        layers = [
            "Hillshade (Vertex Attribute R)",
            "Ambient Occlusion (Vertex Attribute G)",
            "Texture detail (Vertex Attribute B)",
            "Vegetation Cover (Vertex Attribute A)",
            "Landcover Class (Discrete Face Array)",
            "Soil Color Class (Discrete Face Array)"
        ]
        
        for lyr in layers:
            item = QtWidgets.QListWidgetItem(lyr)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.layer_list.addItem(item)
            
        self.layer_list.itemChanged.connect(self.on_layer_selection_mutated)
        layer_layout.addWidget(self.layer_list)
        self.sidebar.layout().addWidget(layer_group)

        # Discrete Palette Editor Box
        self.palette_group = QtWidgets.QGroupBox("Discrete Palette Mapping Manager")
        self.palette_layout = QtWidgets.QVBoxLayout(self.palette_group)
        self.palette_scroll = QtWidgets.QScrollArea()
        self.palette_scroll.setWidgetResizable(True)
        self.palette_container = QtWidgets.QWidget()
        self.palette_container_layout = QtWidgets.QVBoxLayout(self.palette_container)
        self.palette_scroll.setWidget(self.palette_container)
        self.palette_layout.addWidget(self.palette_scroll)
        self.sidebar.layout().addWidget(self.palette_group)
        self.palette_group.setVisible(False)

        # Warp Settings
        warp_group = QtWidgets.QGroupBox("Funnel Geometry Deformation Curve")
        warp_layout = QtWidgets.QVBoxLayout(warp_group)
        warp_layout.addWidget(QtWidgets.QLabel("Warp Amplitude:"))
        self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.amp_slider.setRange(0, 2000)
        self.amp_slider.valueChanged.connect(self.update_pipeline)
        warp_layout.addWidget(self.amp_slider)
        
        warp_layout.addWidget(QtWidgets.QLabel("Decay k-Factor:"))
        self.decay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.decay_slider.setRange(1, 150)
        self.decay_slider.setValue(25)
        self.decay_slider.valueChanged.connect(self.update_pipeline)
        warp_layout.addWidget(self.decay_slider)
        self.sidebar.layout().addWidget(warp_group)

        # Camera settings
        cam_group = QtWidgets.QGroupBox("Locked Camera Focal Coordinates")
        cam_layout = QtWidgets.QVBoxLayout(cam_group)
        self.alt_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alt_slider.setRange(1, 2000)
        self.alt_slider.valueChanged.connect(self.update_camera_orientation)
        cam_layout.addWidget(QtWidgets.QLabel("Altitude Height:"))
        cam_layout.addWidget(self.alt_slider)
        self.sidebar.layout().addWidget(cam_group)

    def on_load_mesh_clicked(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Target Geometry PLY", "", "PLY Meshes (*.ply)")
        if not file_path: return
        try:
            if self.mesh_actor is not None: self.plotter.remove_actor(self.mesh_actor)
            
            self.mesh = pv.read(file_path)
            
            # Read hidden face-level custom array fields using plyfile structure blocks
            ply_raw = PlyData.read(file_path)
            if 'landcover' in ply_raw['face'].data.dtype.names:
                self.mesh.cell_data['landcover'] = np.asarray(ply_raw['face']['landcover']).astype(np.int32)
                self.mesh.cell_data['soil_color'] = np.asarray(ply_raw['face']['soil_color']).astype(np.int32)

            # Extract distinct unique IDs present in dataset to initialize palette engine hooks
            u_lc = np.unique(self.mesh.cell_data['landcover']) if 'landcover' in self.mesh.cell_data else [0,1,2,3,4]
            for val in u_lc:
                if val not in self.lc_colors: self.lc_colors[int(val)] = "#71717a"
            
            u_soil = np.unique(self.mesh.cell_data['soil_color']) if 'soil_color' in self.mesh.cell_data else [0,1,2,3]
            for val in u_soil:
                if val not in self.soil_colors: self.soil_colors[int(val)] = "#a1a1aa"

            total_pts = self.mesh.n_points
            step = max(1, total_pts // 40000)
            self.lightweight_points = np.asarray(self.mesh.points[::step, :].copy())
            self.minimap.compute_dem_from_downsampled(self.lightweight_points)

            xmin, xmax, ymin, ymax, _, _ = self.mesh.bounds
            diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            
            self._block_updates = True
            self.amp_slider.setRange(0, int(diagonal * 1.5))
            self.amp_slider.setValue(int(diagonal * 0.25))
            self.alt_slider.setRange(1, int(diagonal * 0.8))
            self.alt_slider.setValue(int(diagonal * 0.12))
            
            # Force first option default layout to initialize pipeline safely
            self.layer_list.item(0).setCheckState(QtCore.Qt.Checked)
            self._block_updates = False

            self.rebuild_mesh_actor()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Failure", f"Failed parsing packed PLY architecture:\n{str(e)}")

    def on_layer_selection_mutated(self, item):
        if self._block_updates or self.mesh is None: return
        self._block_updates = True
        
        # Enforce strict single-layer visibility mutex behavior
        if item.checkState() == QtCore.Qt.Checked:
            for i in range(self.layer_list.count()):
                other_item = self.layer_list.item(i)
                if other_item is not item:
                    other_item.setCheckState(QtCore.Qt.Unchecked)
        
        self._block_updates = False
        self.rebuild_mesh_actor()

    def get_active_layer_index(self):
        for i in range(self.layer_list.count()):
            if self.layer_list.item(i).checkState() == QtCore.Qt.Checked:
                return i
        return 0

    def populate_palette_editor(self, active_idx):
        # Clear editing layout container
        while self.palette_container_layout.count():
            w = self.palette_container_layout.takeAt(0).widget()
            if w: w.deleteLater()

        if active_idx not in [4, 5]:
            self.palette_group.setVisible(False)
            return

        self.palette_group.setVisible(True)
        working_palette = self.lc_colors if active_idx == 4 else self.soil_colors
        label_prefix = "Landcover" if active_idx == 4 else "Soil Class"

        for cid, color_hex in working_palette.items():
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            
            lbl = QtWidgets.QLabel(f"{label_prefix} ID {cid}:")
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(50)
            btn.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #ffffff;")
            
            # Use inline cell lambda block capturing to track state
            btn.clicked.connect(lambda checked, c=cid, b=btn, idx=active_idx: self.open_color_picker(idx, c, b))
            
            row_layout.addWidget(lbl)
            row_layout.addWidget(btn)
            self.palette_container_layout.addWidget(row)

    def open_color_picker(self, layer_idx, class_id, button_widget):
        palette_target = self.lc_colors if layer_idx == 4 else self.soil_colors
        init_color = QtGui.QColor(palette_target[class_id])
        chosen = QtWidgets.QColorDialog.getColor(init_color, self, "Select Class Hex Value Mapping")
        if chosen.isValid():
            palette_target[class_id] = chosen.name()
            button_widget.setStyleSheet(f"background-color: {chosen.name()}; border: 1px solid #ffffff;")
            self.rebuild_mesh_actor()

    def rebuild_mesh_actor(self):
        if self.mesh is None: return
        
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)

        active_idx = self.get_active_layer_index()
        self.populate_palette_editor(active_idx)

        # 1. Processing and mapping continuous point fields (Hillshade / AO / Details / Veg)
        if active_idx in [0, 1, 2, 3]:
            rgba_array = np.asarray(self.mesh.point_data["RGBA"])
            extracted_band = rgba_array[:, active_idx].astype(np.float32)
            self.mesh.point_data["RenderActive"] = extracted_band
            
            # Map single channel continuous metrics to gray palettes
            self.mesh_actor = self.plotter.add_mesh(
                self.mesh, scalars="RenderActive", cmap="gray", 
                show_scalar_bar=False, lighting=False
            )
        
        # 2. Map Categorical Polygons (Discrete Landcover / Soil palettes)
        elif active_idx in [4, 5]:
            target_key = "landcover" if active_idx == 4 else "soil_color"
            palette_map = self.lc_colors if active_idx == 4 else self.soil_colors
            
            cell_ids = self.mesh.cell_data[target_key]
            max_id = max(palette_map.keys()) if len(palette_map) > 0 else 10
            
            # Build lookup array mapping category integer values directly to continuous space
            lookup_table = np.zeros(max_id + 1, dtype=np.float32)
            color_list = []
            
            for idx, cid in enumerate(sorted(palette_map.keys())):
                lookup_table[cid] = float(idx)
                color_list.append(palette_map[cid])
            
            mapped_scalars = lookup_table[cell_ids]
            self.mesh.cell_data["RenderActiveDiscrete"] = mapped_scalars
            
            self.mesh_actor = self.plotter.add_mesh(
                self.mesh, scalars="RenderActiveDiscrete", cmap=color_list,
                show_scalar_bar=False, lighting=False
            )

        # Re-inject the continuous progressive projection warp replacement code
        sp = self.mesh_actor.GetShaderProperty()
        sp.ClearAllShaderReplacements()

        impl_code = """
            float d = distance(vertexMC.xy, u_focalCenter);
            float normDist = d / u_maxDist;
            float verticalLift = u_amplitude * (1.0 - exp(-u_kDecay * normDist));
            vertexMC.z += verticalLift;
        """
        sp.AddVertexShaderReplacement("//VTK::PositionVC::Impl", False, impl_code, False)
        
        self.on_minimap_center_updated(self.minimap.cx, self.minimap.cy)

    def on_minimap_center_updated(self, cx, cy):
        if self.mesh is None: return
        mx, my = self.minimap.get_mesh_coords()
        self.update_pipeline()

    def update_pipeline(self):
        if self.mesh is None or self.mesh_actor is None or self._block_updates: return

        amplitude = float(self.amp_slider.value())
        k_decay = float(self.decay_slider.value() / 10.0)
        cx, cy = self.minimap.get_mesh_coords()

        xmin, xmax, ymin, ymax, _, _ = self.mesh.bounds
        corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
        max_dist = float(np.max(np.sqrt((corners[:, 0] - cx)**2 + (corners[:, 1] - cy)**2)))
        if max_dist <= 0: max_dist = 1.0

        shader_params = self.mesh_actor.GetShaderProperty().GetVertexCustomUniforms()
        shader_params.SetUniformf("u_amplitude", amplitude)
        shader_params.SetUniformf("u_kDecay", k_decay)
        shader_params.SetUniform2f("u_focalCenter", (cx, cy))
        shader_params.SetUniformf("u_maxDist", max_dist)

        self.update_camera_orientation()

    def update_camera_orientation(self):
        if self.mesh is None or self._block_updates: return
        cx, cy = self.minimap.get_mesh_coords()
        eye_altitude = self.alt_slider.value()

        x, y = self.lightweight_points[:, 0], self.lightweight_points[:, 1]
        center_idx = np.argmin((x - cx)**2 + (y - cy)**2)
        z_base = self.lightweight_points[center_idx, 2]

        camera_pos = np.array([cx, cy, z_base + eye_altitude])
        focal_point = np.array([cx, cy, z_base])

        self.plotter.camera.position = camera_pos
        self.plotter.camera.focal_point = focal_point
        self.plotter.camera.up = (0.0, 1.0, 0.0)
        self.plotter.render()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())