import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

class MinimapWidget(QtWidgets.QWidget):
    centerChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.cx, self.cy = 0.5, 0.5  
        self.dem_image = None
        self.xmin = self.xmax = self.ymin = self.ymax = 0.0

    def compute_dem_from_downsampled(self, pts_downsampled, active_layer_idx=0):
        if pts_downsampled is None or len(pts_downsampled) == 0: 
            return
        
        self.xmin, self.xmax = pts_downsampled[:, 0].min(), pts_downsampled[:, 0].max()
        self.ymin, self.ymax = pts_downsampled[:, 1].min(), pts_downsampled[:, 1].max()

        # Extract coordinates and active render attribute targets
        x = pts_downsampled[:, 0]
        y = pts_downsampled[:, 1]
        
        if active_layer_idx in [0, 1, 2, 3] and pts_downsampled.shape[1] > 3:
            scalars = pts_downsampled[:, active_layer_idx]
        else:
            scalars = pts_downsampled[:, 2] # Fallback directly to Z elevation

        res = 256
        
        # 1. Create a perfectly uniform grid matching the target image resolution
        grid_x, grid_y = np.mgrid[
            self.xmin:self.xmax:complex(0, res),
            self.ymin:self.ymax:complex(0, res)
        ]

        # 2. Resample using linear interpolation instead of binning to eliminate stripe waves
        stride = max(1, len(pts_downsampled) // 50000)
        points_subset = np.column_stack((x[::stride], y[::stride]))
        scalars_subset = scalars[::stride]

        grid_vals = griddata(
            points_subset, 
            scalars_subset, 
            (grid_x, grid_y), 
            method='linear'
        )

        # Fill any missing corner edge cells cleanly
        nan_mask = np.isnan(grid_vals)
        if np.any(nan_mask):
            grid_vals[nan_mask] = np.nanmean(grid_vals) if not np.all(nan_mask) else 0.0

        # 3. Apply a light smoothing filter to blend any remaining grain variations
        grid_vals = gaussian_filter(grid_vals, sigma=1.2)

        # 4. Map back down into standard 8-bit monochromatic intensity ranges
        v_min, v_max = grid_vals.min(), grid_vals.max()
        if v_max != v_min:
            norm_bytes = ((grid_vals - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
        else:
            norm_bytes = np.zeros((res, res), dtype=np.uint8)

        # Rotate/Transpose to correctly map spatial grid shapes right onto your landscape coordinate orientation
        oriented_bytes = np.ascontiguousarray(np.flipud(norm_bytes.T))
        
        # 5. Let Qt process its internal line stride padding allocations cleanly
        byte_data = QtCore.QByteArray(oriented_bytes.tobytes())
        img = QtGui.QImage.fromData(byte_data)
        
        if img.isNull():
            img = QtGui.QImage(oriented_bytes.data, res, res, res, QtGui.QImage.Format_Grayscale8).copy()
        else:
            img = img.convertToFormat(QtGui.QImage.Format_Grayscale8)

        self.dem_image = img
        self.update()

    def get_mesh_coords(self):
        mx = self.xmin + self.cx * (self.xmax - self.xmin)
        my = self.ymin + self.cy * (self.ymax - self.ymin)
        return mx, my

    def _get_aspect_corrected_rect(self):
        """Helper to compute map geometry framing inside available widget viewport space."""
        pad = 4
        w, h = self.width() - 2*pad, self.height() - 2*pad
        if w <= 0 or h <= 0:
            return 0, 0, 0, 0, 0, 0

        spatial_w = self.xmax - self.xmin
        spatial_h = self.ymax - self.ymin
        aspect_ratio = spatial_h / spatial_w if spatial_w > 0 else 1.0

        if w * aspect_ratio <= h:
            view_w = w
            view_h = int(w * aspect_ratio)
        else:
            view_h = h
            view_w = int(h / aspect_ratio)

        ox = pad + (w - view_w) // 2
        oy = pad + (h - view_h) // 2
        return ox, oy, view_w, view_h

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        
        painter.fillRect(self.rect(), QtGui.QColor("#1e293b"))
        
        ox, oy, view_w, view_h = self._get_aspect_corrected_rect()
        if view_w <= 0 or view_h <= 0: return

        # Draw image within un-distorted boundaries
        if self.dem_image is not None and not self.dem_image.isNull():
            painter.drawImage(QtCore.QRect(ox, oy, view_w, view_h), self.dem_image)
            
        painter.setPen(QtGui.QPen(QtGui.QColor("#334155"), 1))
        painter.drawRect(ox, oy, view_w, view_h)
            
        hx = int(ox + self.cx * view_w)
        hy = int(oy + (1.0 - self.cy) * view_h)
        
        painter.setPen(QtGui.QPen(QtGui.QColor("#ef4444"), 1, QtCore.Qt.DashLine))
        painter.drawLine(ox, hy, ox + view_w, hy)
        painter.drawLine(hx, oy, hx, oy + view_h)
        
        painter.setPen(QtGui.QPen(QtGui.QColor("#f8fafc"), 1.5))
        painter.setBrush(QtGui.QColor("#ef4444"))
        painter.drawEllipse(QtCore.QPoint(hx, hy), 5, 5)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton: 
            self.update_from_mouse(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton: 
            self.update_from_mouse(event.pos())

    def update_from_mouse(self, pos):
        ox, oy, view_w, view_h = self._get_aspect_corrected_rect()
        if view_w <= 0 or view_h <= 0: return
        
        # Evaluate crosshair alignment relative to corrected sub-grid space
        self.cx = np.clip((pos.x() - ox) / view_w, 0.0, 1.0)
        self.cy = np.clip(1.0 - ((pos.y() - oy) / view_h), 0.0, 1.0)
        self.update()
        self.centerChanged.emit(self.cx, self.cy)