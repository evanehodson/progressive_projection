import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

class MinimapWidget(QtWidgets.QWidget):
    centerChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.cx, self.cy = 0.5, 0.5  
        self.dem_image = None
        self.xmin = self.xmax = self.ymin = self.ymax = 0.0

    def compute_dem_from_downsampled(self, pts_downsampled):
        if pts_downsampled is None or len(pts_downsampled) == 0: return
        
        self.xmin, self.xmax = pts_downsampled[:, 0].min(), pts_downsampled[:, 0].max()
        self.ymin, self.ymax = pts_downsampled[:, 1].min(), pts_downsampled[:, 1].max()

        res = 128 
        grid_z = np.zeros((res, res), dtype=np.float32)
        grid_counts = np.zeros((res, res), dtype=np.float32)

        x_idx = np.clip(((pts_downsampled[:, 0] - self.xmin) / (self.xmax - self.xmin) * (res - 1)).astype(np.int32), 0, res - 1)
        y_idx = np.clip(((pts_downsampled[:, 1] - self.ymin) / (self.ymax - self.ymin) * (res - 1)).astype(np.int32), 0, res - 1)

        for i in range(len(pts_downsampled)):
            grid_z[y_idx[i], x_idx[i]] += pts_downsampled[i, 2]
            grid_counts[y_idx[i], x_idx[i]] += 1.0

        valid = grid_counts > 0
        grid_z[valid] /= grid_counts[valid]
        if not np.all(valid):
            grid_z[~valid] = grid_z[valid].mean() if np.any(valid) else 0.0

        z_min, z_max = grid_z.min(), grid_z.max()
        if z_max != z_min:
            norm_z = ((grid_z - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)
        else:
            norm_z = np.zeros((res, res), dtype=np.uint8)

        self.dem_image = QtGui.QImage(res, res, QtGui.QImage.Format_Grayscale8)
        flipped_z = np.flipud(norm_z)
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
            painter.drawImage(QtCore.QRect(pad, pad, w, h), self.dem_image)
            
        hx, hy = int(pad + self.cx * w), int(pad + (1.0 - self.cy) * h)
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