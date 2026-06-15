import random
from PyQt5 import QtWidgets, QtCore, QtGui

class ColorButton(QtWidgets.QPushButton):
    """A flat colored square that triggers a QColorDialog."""
    colorChanged = QtCore.pyqtSignal(int, QtGui.QColor)

    def __init__(self, class_id, color, parent=None):
        super().__init__(parent)
        self.class_id = class_id
        self._color = color
        self.setFixedSize(18, 18)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.update_stylesheet()
        self.clicked.connect(self.choose_color)

    def update_stylesheet(self):
        self.setStyleSheet(f"""
            background-color: {self._color.name()}; 
            border: 1px solid #475569; 
            border-radius: 3px;
        """)

    def set_color(self, qcolor):
        self._color = qcolor
        self.update_stylesheet()

    def choose_color(self):
        color = QtWidgets.QColorDialog.getColor(
            self._color, 
            self, 
            f"Select Color for Class {self.class_id}"
        )
        if color.isValid():
            self._color = color
            self.update_stylesheet()
            self.colorChanged.emit(self.class_id, color)


class GISLayerTreeWidget(QtWidgets.QTreeWidget):
    """
    An interactive GIS Legend Panel. Enforces mutual exclusivity, 
    auto-collapses inactive layers, and supports live color modifications
    for both continuous and categorical rendering channels.
    """
    layerChanged = QtCore.pyqtSignal(int)

    def __init__(self, pipeline, plotter, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.plotter = plotter
        
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setAnimated(True)
        self.setStyleSheet("""
            QTreeWidget { background-color: #1e293b; color: #f8fafc; border: none; font-weight: bold; }
            QTreeWidget::item { padding: 4px; }
            QTreeWidget::item:selected { background-color: transparent; }
            QTreeWidget::indicator:checked { background-color: #10b981; border: 1px solid #cbd5e1; border-radius: 2px; }
            QTreeWidget::indicator:unchecked { background-color: #334155; border: 1px solid #cbd5e1; border-radius: 2px; }
        """)

        # Stripped parentheticals down to clean geographic layer titles
        self.layers = [
            ("Hillshade", 0),
            ("Ambient Occlusion", 1),
            ("Texture Detail", 2),
            ("Vegetation Cover", 3),
            ("Landcover Class", 4),
            ("Soil Color Class", 5)
        ]
        
        # Tracks active color ramp names for each continuous point attribute layer
        self.continuous_ramps = {0: "Grayscale", 1: "Grayscale", 2: "Grayscale", 3: "Grayscale"}
        
        self.nlcd_classes = {
            11: "Open Water", 12: "Perennial Ice/Snow", 21: "Developed, Open Space",
            22: "Developed, Low Intensity", 23: "Developed, Medium Intensity",
            24: "Developed, High Intensity", 31: "Barren Land", 41: "Deciduous Forest",
            42: "Evergreen Forest", 43: "Mixed Forest", 52: "Shrub/Scrub",
            71: "Grassland/Herbaceous", 81: "Pasture/Hay", 82: "Cultivated Crops",
            90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands"
        }

        self.top_items = []
        self._build_tree()
        
        self.itemClicked.connect(self.on_item_clicked)

    def _build_tree(self):
        """Generates root radio selection rows and populates default structural children."""
        for name, idx in self.layers:
            item = QtWidgets.QTreeWidgetItem(self)
            item.setText(0, name)
            item.setData(0, QtCore.Qt.UserRole, idx)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            
            # Start up default selection state matching the engine initialization
            if idx == 0:
                item.setCheckState(0, QtCore.Qt.Checked)
                item.setExpanded(True)
                self._populate_continuous_controls(item, idx)
            else:
                item.setCheckState(0, QtCore.Qt.Unchecked)
                
            self.top_items.append(item)

    def on_item_clicked(self, item, column):
        """Handles accordion style auto-expansion and applies real-time engine changes."""
        idx = item.data(0, QtCore.Qt.UserRole)
        if idx is None:
            return  # Skip child interaction rows

        self.blockSignals(True)
        for top_item in self.top_items:
            if top_item != item:
                top_item.setCheckState(0, QtCore.Qt.Unchecked)
                top_item.setExpanded(False)
        item.setCheckState(0, QtCore.Qt.Checked)
        item.setExpanded(True)
        self.blockSignals(False)
        
        # 1. Fire baseline layer change to update backend state machine layout
        self.layerChanged.emit(idx)
        
        # 2. Lazy load or apply custom hardware overrides
        if idx in [0, 1, 2, 3]:
            self._populate_continuous_controls(item, idx)
            self._apply_continuous_ramp(idx, self.continuous_ramps[idx])
        elif idx == 4:
            self._populate_nlcd(item)
        elif idx == 5:
            self._populate_soil(item)

    # -------------------------------------------------------------------------
    # CONTINUOUS RAMP MANAGEMENT MODULE (Layers 0-3)
    # -------------------------------------------------------------------------
    def _populate_continuous_controls(self, parent_item, layer_idx):
        """Renders an elegant dropdown menu for selecting continuous color gradients."""
        if parent_item.childCount() > 0:
            return
            
        child = QtWidgets.QTreeWidgetItem(parent_item)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(16, 2, 0, 2)
        
        lbl = QtWidgets.QLabel("Color Ramp:")
        lbl.setStyleSheet("color: #94a3b8; font-weight: normal;")
        
        combo = QtWidgets.QComboBox()
        combo.addItems(["Grayscale", "Viridis", "Plasma", "Terrain"])
        combo.setCurrentText(self.continuous_ramps[layer_idx])
        combo.currentTextChanged.connect(lambda text: self._on_ramp_changed(layer_idx, text))
        
        layout.addWidget(lbl)
        layout.addWidget(combo)
        layout.addStretch()
        self.setItemWidget(child, 0, widget)

    def _on_ramp_changed(self, layer_idx, ramp_name):
        self.continuous_ramps[layer_idx] = ramp_name
        self._apply_continuous_ramp(layer_idx, ramp_name)

    def _apply_continuous_ramp(self, layer_idx, ramp_name):
        """Overwrites the transient grayscale VTK table on the fly."""
        if not self.pipeline.mesh_actor: 
            return
        mapper = self.pipeline.mesh_actor.GetMapper()
        lut = mapper.GetLookupTable()
        if not lut: 
            return

        colors = self._interpolate_ramp_data(ramp_name, steps=256)
        lut.SetNumberOfTableValues(256)
        for i, (r, g, b) in enumerate(colors):
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Modified()
        self.plotter.render()

    def _interpolate_ramp_data(self, ramp_name, steps=256):
        """Mathematical multi-point anchor interpolation for hardware gradients."""
        ramps = {
            "Grayscale": [(0.0, (0,0,0)), (1.0, (255,255,255))],
            "Viridis":   [(0.0, (68,1,84)), (0.25, (59,82,139)), (0.5, (33,144,141)), (0.75, (94,201,98)), (1.0, (253,231,37))],
            "Plasma":    [(0.0, (13,8,135)), (0.25, (126,3,168)), (0.5, (204,71,120)), (0.75, (248,149,64)), (1.0, (240,249,33))],
            "Terrain":   [(0.0, (0,0,128)), (0.15, (0,128,255)), (0.25, (240,240,64)), (0.5, (32,160,32)), (0.75, (128,64,0)), (1.0, (255,255,255))]
        }
        anchors = ramps.get(ramp_name, ramps["Grayscale"])
        colors = []
        for i in range(steps):
            t = i / (steps - 1)
            for idx in range(len(anchors) - 1):
                t0, c0 = anchors[idx]
                t1, c1 = anchors[idx+1]
                if t0 <= t <= t1 or idx == len(anchors) - 2:
                    denom = (t1 - t0) if (t1 - t0) != 0 else 1.0
                    f = (t - t0) / denom
                    r = c0[0] + f * (c1[0] - c0[0])
                    g = c0[1] + f * (c1[1] - c0[1])
                    b = c0[2] + f * (c1[2] - c0[2])
                    colors.append((r/255.0, g/255.0, b/255.0))
                    break
        return colors

    # -------------------------------------------------------------------------
    # CATEGORICAL SYMBOLOGY TOOLSETS (Layers 4-5)
    # -------------------------------------------------------------------------
    def _create_global_toolbar(self, parent_item, layer_idx):
        """Creates batch manipulation action triggers (Randomize / Contrasting Palette)."""
        child = QtWidgets.QTreeWidgetItem(parent_item)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(14, 2, 0, 4)
        
        btn_rand = QtWidgets.QPushButton("Randomize")
        btn_dist = QtWidgets.QPushButton("Distinct Set")
        
        btn_style = """
            QPushButton { background-color: #334155; color: #f1f5f9; border: 1px solid #475569; border-radius: 3px; font-size: 7.5pt; padding: 3px 8px; font-weight: normal; }
            QPushButton::hover { background-color: #475569; }
        """
        btn_rand.setStyleSheet(btn_style)
        btn_dist.setStyleSheet(btn_style)
        
        btn_rand.clicked.connect(lambda: self._batch_modify_categorical(parent_item, layer_idx, "random"))
        btn_dist.clicked.connect(lambda: self._batch_modify_categorical(parent_item, layer_idx, "distinct"))
        
        layout.addWidget(btn_rand)
        layout.addWidget(btn_dist)
        layout.addStretch()
        self.setItemWidget(child, 0, widget)

    def _batch_modify_categorical(self, parent_item, layer_idx, method):
        """Generates algorithmic or randomized adjustments across entire tables."""
        lut = self.pipeline.nlcd_lut if layer_idx == 4 else self.pipeline.soil_lut
        valid_codes = list(self.nlcd_classes.keys()) if layer_idx == 4 else self._get_active_soil_codes()
        
        for i, code in enumerate(valid_codes):
            if method == "random":
                r, g, b = random.random(), random.random(), random.random()
            else:
                # Golden-ratio / spacing technique around the HSV wheel for max contrast separation
                hue = i / len(valid_codes)
                sat = 0.85 if (i % 2 == 0) else 0.55
                val = 0.90 if (i % 3 != 0) else 0.65
                qc = QtGui.QColor.fromHsvF(hue, sat, val)
                r, g, b = qc.red()/255.0, qc.green()/255.0, qc.blue()/255.0
                
            lut.SetTableValue(code, r, g, b, 1.0)
            
        lut.Modified()
        self.plotter.render()
        
        # Flush and reconstruct sub-branches to update UI colors immediately
        if layer_idx == 4:
            self._populate_nlcd(parent_item, force=True)
        else:
            self._populate_soil(parent_item, force=True)

    def _populate_nlcd(self, parent_item, force=False):
        """Extracts and monitors NLCD category array definitions directly from hardware."""
        if not force and parent_item.childCount() > 0: 
            return
        parent_item.takeChildren()
        self._create_global_toolbar(parent_item, 4)
        
        for code, name in self.nlcd_classes.items():
            child = QtWidgets.QTreeWidgetItem(parent_item)
            rgba = [0.0, 0.0, 0.0, 1.0]
            self.pipeline.nlcd_lut.GetTableValue(code, rgba)
            color = QtGui.QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            
            widget = self._create_legend_row(code, f"Class {code}: {name}", color, self.on_nlcd_color_changed)
            self.setItemWidget(child, 0, widget)

    def _populate_soil(self, parent_item, force=False):
        """Scans the active soil index space, ignoring background/safety mask slots."""
        if not force and parent_item.childCount() > 0: 
            return
        parent_item.takeChildren()
        self._create_global_toolbar(parent_item, 5)
        
        valid_codes = self._get_active_soil_codes()
        for code in valid_codes:
            child = QtWidgets.QTreeWidgetItem(parent_item)
            rgba = [0.0, 0.0, 0.0, 1.0]
            self.pipeline.soil_lut.GetTableValue(code, rgba)
            color = QtGui.QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            
            widget = self._create_legend_row(code, f"Soil Unit {code}", color, self.on_soil_color_changed)
            self.setItemWidget(child, 0, widget)

    def _get_active_soil_codes(self):
        """Helper to collect actual soil codes by ignoring background slate presets (0.46, 0.41, 0.37)."""
        valid_codes = []
        num_colors = self.pipeline.soil_lut.GetNumberOfTableValues()
        for code in range(num_colors):
            rgba = [0.0, 0.0, 0.0, 1.0]
            self.pipeline.soil_lut.GetTableValue(code, rgba)
            r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            # Filter out baseline fill tones or safety bounds masks
            if (r == 117 and g == 104 and b == 94) or (r == 30 and g == 35 and b == 45):
                continue
            valid_codes.append(code)
        return valid_codes

    def _create_legend_row(self, code, text, color, slot_callback):
        """Assembles a clean, flat layout containing the color swatch and description."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(16, 2, 0, 2)
        
        btn = ColorButton(code, color)
        btn.colorChanged.connect(slot_callback)
        
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #cbd5e1; font-weight: normal; font-size: 8pt;")
        
        layout.addWidget(btn)
        layout.addWidget(lbl)
        layout.addStretch()
        return widget

    def on_nlcd_color_changed(self, class_id, qcolor):
        if not self.pipeline.mesh_actor: return
        self.pipeline.nlcd_lut.SetTableValue(
            class_id, qcolor.red() / 255.0, qcolor.green() / 255.0, qcolor.blue() / 255.0, 1.0
        )
        self.plotter.render()

    def on_soil_color_changed(self, class_id, qcolor):
        if not self.pipeline.mesh_actor: return
        self.pipeline.soil_lut.SetTableValue(
            class_id, qcolor.red() / 255.0, qcolor.green() / 255.0, qcolor.blue() / 255.0, 1.0
        )
        self.plotter.render()