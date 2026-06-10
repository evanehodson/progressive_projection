from PyQt5 import QtWidgets, QtCore

class DeformationControls(QtWidgets.QWidget):
    valuesChanged = QtCore.pyqtSignal(float, float, float) # Amplitude, Decay, Altitude

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Warp Amplitude
        layout.addWidget(QtWidgets.QLabel("Warp Amplitude:"))
        self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.amp_slider.setRange(0, 2000)
        self.amp_slider.valueChanged.connect(self._emit_changes)
        layout.addWidget(self.amp_slider)

        # Decay Factor
        layout.addWidget(QtWidgets.QLabel("Decay k-Factor:"))
        self.decay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.decay_slider.setRange(1, 150)
        self.decay_slider.setValue(25)
        self.decay_slider.valueChanged.connect(self._emit_changes)
        layout.addWidget(self.decay_slider)

        # Camera Altitude
        layout.addWidget(QtWidgets.QLabel("Camera Altitude Height:"))
        self.alt_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alt_slider.setRange(1, 2000)
        self.alt_slider.valueChanged.connect(self._emit_changes)
        layout.addWidget(self.alt_slider)

    def calibrate_ranges(self, diagonal):
        self.amp_slider.setRange(0, int(diagonal * 1.5))
        self.amp_slider.setValue(int(diagonal * 0.25))
        self.alt_slider.setRange(1, int(diagonal * 0.8))
        self.alt_slider.setValue(int(diagonal * 0.12))

    def _emit_changes(self):
        amp = float(self.amp_slider.value())
        decay = float(self.decay_slider.value() / 10.0)
        alt = float(self.alt_slider.value())
        self.valuesChanged.emit(amp, decay, alt)