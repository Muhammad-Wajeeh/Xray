import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from phantom import create_shepp_logan
from simulate_xray import simulate_projection_angle
from simulate_xray import simulate_xray_2d



class XrayGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Simulation GUI")

        # Load phantom once
        self.phantom = create_shepp_logan()

        # Matplotlib figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # --- Create sliders ---
        self.angle_slider = self.create_slider(0, 180, 30, "Angle (deg)")
        self.sid_slider   = self.create_slider(200, 1200, 500, "SID")
        self.sdd_slider   = self.create_slider(400, 1600, 1000, "SDD")
        self.kvp_slider   = self.create_slider(20, 120, 30, "kVp")
        self.exp_slider   = self.create_slider(1, 300, 100, "Exposure x0.01 s")
        self.filt_slider  = self.create_slider(0, 10, 2, "Filtration (mm Al)")

        sliders = QVBoxLayout()
        for label, slider in [
            self.angle_slider,
            self.sid_slider,
            self.sdd_slider,
            self.kvp_slider,
            self.exp_slider,
            self.filt_slider,
        ]:
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(slider)
            sliders.addLayout(row)

        slider_panel = QWidget()
        slider_panel.setLayout(sliders)

        # --- Layout: Matplotlib + sliders ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas, stretch=3)
        main_layout.addWidget(slider_panel, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect slider signals
        for _, slider in [
            self.angle_slider,
            self.sid_slider,
            self.sdd_slider,
            self.kvp_slider,
            self.exp_slider,
            self.filt_slider,
        ]:
            slider.valueChanged.connect(self.update_projection)

        # Initial draw
        self.update_projection()

    def create_slider(self, min_val, max_val, init, text):
        label = QLabel(f"{text}: {init}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init)
        slider.text = text
        return label, slider

    def update_projection(self):
        # Read slider values
        angle = self.angle_slider[1].value()
        sid   = self.sid_slider[1].value()
        sdd   = self.sdd_slider[1].value()
        kvp   = self.kvp_slider[1].value()
        exposure = self.exp_slider[1].value() / 100.0
        filt  = self.filt_slider[1].value()

        # Update labels
        self.angle_slider[0].setText(f"Angle: {angle}°")
        self.sid_slider[0].setText(f"SID: {sid}")
        self.sdd_slider[0].setText(f"SDD: {sdd}")
        self.kvp_slider[0].setText(f"kVp: {kvp}")
        self.exp_slider[0].setText(f"Exposure x0.01s: {self.exp_slider[1].value()}")
        self.filt_slider[0].setText(f"Filtration (mm Al): {filt}")

        projection_img = simulate_xray_2d(
        self.phantom,
        angle,
        I0=1.0,
        sid=sid,
        sdd=sdd,
        kVp=kvp,
        exposure_time=exposure,
        filtration_mmAl=filt)
 
        self.ax.clear()
        self.ax.imshow(projection_img, cmap='gray')
        self.ax.set_title("X-ray Projection (2D Radiograph)")
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    gui = XrayGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
