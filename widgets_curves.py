from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np


class WarpCurveEditor(QtWidgets.QWidget):
    curveChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, max_raise=50.0):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)

        self._max_y = float(max_raise)
        self._points = [(0.0, 0.0), (0.35, 0.04), (1.0, 0.2)]
        self._drag_idx = -1
        self._sample_count = 256
        self._hover_idx = -1

        self._pad_left = 45
        self._pad_right = 12
        self._pad_top = 15
        self._pad_bottom = 28

    def set_max_raise(self, val):
        self._max_y = max(float(val), 1.0)
        self.update()
        self._emit_curve()

    def get_control_points(self):
        return list(self._points)

    def set_control_points(self, pts):
        self._points = [(float(x), float(y)) for x, y in pts]
        self.update()
        self._emit_curve()

    def _to_widget(self, nx, ny):
        w = self.width() - self._pad_left - self._pad_right
        h = self.height() - self._pad_top - self._pad_bottom
        return self._pad_left + nx * w, self._pad_top + (1.0 - ny) * h

    def _from_widget(self, px, py):
        w = self.width() - self._pad_left - self._pad_right
        h = self.height() - self._pad_top - self._pad_bottom
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        nx = (px - self._pad_left) / w
        ny = 1.0 - (py - self._pad_top) / h
        return np.clip(nx, 0.0, 1.0), np.clip(ny, 0.0, 1.0)

    def _interpolate(self, xs, pts):
        """Monotone cubic Hermite with forced 0 slope at both ends (S-curve)."""
        n = len(pts)
        if n < 2:
            return np.zeros_like(xs)
        if n == 2:
            return np.interp(xs, pts[:, 0], pts[:, 1])

        # Slopes: 0 at ends, Fritsch-Carlson for interior
        slopes = np.zeros(n)
        for i in range(1, n - 1):
            h1 = pts[i, 0] - pts[i-1, 0]
            h2 = pts[i+1, 0] - pts[i, 0]
            d1 = (pts[i, 1] - pts[i-1, 1]) / h1
            d2 = (pts[i+1, 1] - pts[i, 1]) / h2
            if d1 * d2 <= 0:
                slopes[i] = 0.0
            else:
                w1 = h2 / (h1 + h2)
                w2 = h1 / (h1 + h2)
                slopes[i] = 1.0 / (w1 / d1 + w2 / d2)

        # Cubic Hermite segment evaluation
        result = np.zeros_like(xs)
        for i in range(n - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            s0, s1 = slopes[i], slopes[i+1]
            dx = x1 - x0
            if dx <= 0:
                continue
            mask = (xs >= x0) & (xs <= x1)
            t = (xs[mask] - x0) / dx
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            result[mask] = h00 * y0 + h10 * s0 * dx + h01 * y1 + h11 * s1 * dx
        return result

    def _sample_curve(self):
        xs = np.linspace(0.0, 1.0, self._sample_count)
        pts = np.array(self._points, dtype=np.float64)
        return self._interpolate(xs, pts)

    def _emit_curve(self):
        ys = self._sample_curve()
        self.curveChanged.emit(ys * self._max_y)

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor("#0f172a"))

        w = self.width() - self._pad_left - self._pad_right
        h = self.height() - self._pad_top - self._pad_bottom
        if w <= 0 or h <= 0:
            qp.end()
            return

        bg = QtGui.QColor("#1e293b")
        grid_pen = QtGui.QPen(QtGui.QColor("#334155"), 1)
        axis_pen = QtGui.QPen(QtGui.QColor("#475569"), 2)
        label_color = QtGui.QColor("#94a3b8")

        qp.fillRect(self._pad_left, self._pad_top, w, h, bg)

        # Grid
        qp.setPen(grid_pen)
        for i in range(1, 5):
            x = self._pad_left + (i / 4.0) * w
            qp.drawLine(int(x), self._pad_top, int(x), self._pad_top + h)
            y = self._pad_top + (i / 4.0) * h
            qp.drawLine(self._pad_left, int(y), self._pad_left + w, int(y))

        # Axes
        qp.setPen(axis_pen)
        qp.drawLine(self._pad_left, self._pad_top, self._pad_left, self._pad_top + h)
        qp.drawLine(self._pad_left, self._pad_top + h, self._pad_left + w, self._pad_top + h)

        # Labels
        font = qp.font()
        font.setPointSize(7)
        qp.setFont(font)
        qp.setPen(label_color)
        qp.drawText(0, self._pad_top, self._pad_left - 4, 14,
                     QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom,
                     f"{self._max_y:.0f}")
        qp.drawText(0, self._pad_top + h // 2 - 7, self._pad_left - 4, 14,
                     QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                     f"{self._max_y / 2:.0f}")
        qp.drawText(0, self._pad_top + h - 12, self._pad_left - 4, 14,
                     QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom, "0")
        qp.drawText(self._pad_left, self._pad_top + h + 4, 50, 14,
                     QtCore.Qt.AlignLeft, "Horizon")
        qp.drawText(self._pad_left + w - 65, self._pad_top + h + 4, 65, 14,
                     QtCore.Qt.AlignRight, "Foreground")

        # Curve
        ys = self._sample_curve()
        xs = np.linspace(0.0, 1.0, self._sample_count)
        path = QtGui.QPainterPath()
        px0, py0 = self._to_widget(xs[0], ys[0])
        path.moveTo(px0, py0)
        for i in range(1, self._sample_count):
            px, py = self._to_widget(xs[i], ys[i])
            path.lineTo(px, py)
        qp.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 2))
        qp.setBrush(QtCore.Qt.NoBrush)
        qp.drawPath(path)

        # Fill under curve
        fill_path = QtGui.QPainterPath()
        px0, py0 = self._to_widget(xs[0], ys[0])
        fill_path.moveTo(self._pad_left, self._pad_top + h)
        fill_path.lineTo(px0, py0)
        for i in range(1, self._sample_count):
            px, py = self._to_widget(xs[i], ys[i])
            fill_path.lineTo(px, py)
        fill_path.lineTo(self._pad_left + w, self._pad_top + h)
        fill_path.closeSubpath()
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(16, 185, 129, 40))
        qp.drawPath(fill_path)

        # Control points
        for i, (x, y) in enumerate(self._points):
            px, py = self._to_widget(x, y)
            if i == 0:
                color = QtGui.QColor("#94a3b8")
                qp.setPen(QtGui.QPen(color, 2))
                qp.setBrush(QtGui.QBrush(color))
                qp.drawEllipse(QtCore.QPointF(px, py), 4, 4)
            elif i == self._drag_idx or i == self._hover_idx:
                color = QtGui.QColor("#fbbf24")
                qp.setPen(QtGui.QPen(color, 2))
                qp.setBrush(QtGui.QBrush(color))
                qp.drawEllipse(QtCore.QPointF(px, py), 6, 6)
            else:
                color = QtGui.QColor("#60a5fa")
                qp.setPen(QtGui.QPen(color, 2))
                qp.setBrush(QtGui.QBrush(color))
                qp.drawEllipse(QtCore.QPointF(px, py), 4, 4)

        qp.end()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            for i, (x, y) in enumerate(self._points):
                if i == 0:
                    continue
                px, py = self._to_widget(x, y)
                dx = event.pos().x() - px
                dy = event.pos().y() - py
                if dx * dx + dy * dy < 100:
                    self._drag_idx = i
                    return

    def mouseMoveEvent(self, event):
        if self._drag_idx >= 0:
            xn, yn = self._from_widget(event.pos().x(), event.pos().y())
            xn = np.clip(xn, 0.0, 1.0)
            yn = np.clip(yn, 0.0, 1.0)
            if self._drag_idx == len(self._points) - 1:
                xn = 1.0
            elif self._drag_idx > 0:
                xn = max(xn, self._points[self._drag_idx - 1][0] + 0.005)
            if self._drag_idx < len(self._points) - 1:
                xn = min(xn, self._points[self._drag_idx + 1][0] - 0.005)
            self._points[self._drag_idx] = (xn, yn)
            self.update()
            self._emit_curve()
        else:
            self._hover_idx = -1
            for i, (x, y) in enumerate(self._points):
                if i == 0:
                    continue
                px, py = self._to_widget(x, y)
                dx = event.pos().x() - px
                dy = event.pos().y() - py
                if dx * dx + dy * dy < 100:
                    self._hover_idx = i
                    break
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_idx = -1

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            xn, yn = self._from_widget(event.pos().x(), event.pos().y())
            if 0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0:
                self._points.append((xn, yn))
                self._points.sort(key=lambda p: p[0])
                self.update()
                self._emit_curve()

    def contextMenuEvent(self, event):
        for i, (x, y) in enumerate(self._points):
            if i == 0 or i == len(self._points) - 1:
                continue
            px, py = self._to_widget(x, y)
            dx = event.pos().x() - px
            dy = event.pos().y() - py
            if dx * dx + dy * dy < 100 and len(self._points) > 2:
                self._points.pop(i)
                self.update()
                self._emit_curve()
                return


class DeformationControls(QtWidgets.QWidget):
    warpProfileChanged = QtCore.pyqtSignal(object)
    viewAngleChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.curve_editor = WarpCurveEditor()
        self.curve_editor.curveChanged.connect(self.warpProfileChanged.emit)
        layout.addWidget(self.curve_editor)

        layout.addWidget(QtWidgets.QLabel("View Direction (Azimuth):"))
        self.view_angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.view_angle_slider.setRange(0, 359)
        self.view_angle_slider.setValue(0)
        self.view_angle_slider.valueChanged.connect(self._on_view_angle_changed)
        layout.addWidget(self.view_angle_slider)

        self.view_angle_label = QtWidgets.QLabel("0\u00b0 (North)")
        layout.addWidget(self.view_angle_label)

        layout.addWidget(QtWidgets.QLabel("Camera Altitude Height:"))
        self.alt_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alt_slider.setRange(1, 2000)
        layout.addWidget(self.alt_slider)

    def _on_view_angle_changed(self, val):
        labels = {0: "North", 90: "East", 180: "South", 270: "West"}
        label = labels.get(val, "")
        if label:
            self.view_angle_label.setText(f"{val}\u00b0 ({label})")
        else:
            self.view_angle_label.setText(f"{val}\u00b0")
        self.viewAngleChanged.emit(float(val))

    def calibrate_ranges(self, diagonal):
        self.alt_slider.setRange(1, int(diagonal * 0.8))
        self.alt_slider.setValue(int(diagonal * 0.12))
        self.curve_editor.set_max_raise(diagonal * 0.25)
