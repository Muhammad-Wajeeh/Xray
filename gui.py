import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from phantom import create_shepp_logan
from simulate_xray import (
    simulate_sinogram,
    simulate_projection_single,
    simulate_projection,
    simulate_projection_angle,
)


class XrayGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Simulation GUI")

        # Load phantom once
        self.phantom = create_shepp_logan()

        # -----------------------
        # Matplotlib figure (image + profile overlays)
        # -----------------------
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax_img, self.ax_profile = self.fig.subplots(2, 1)

        # -----------------------
        # Create sliders
        # -----------------------
        self.angle_slider = self.create_slider(0, 180, 30, "Angle (deg)")
        self.sid_slider   = self.create_slider(200, 1200, 500, "SID")
        self.sdd_slider   = self.create_slider(400, 1600, 1000, "SDD")
        self.kvp_slider   = self.create_slider(20, 120, 30, "kVp")
        self.exp_slider   = self.create_slider(1, 300, 100, "Exposure x0.01 s")
        self.filt_slider  = self.create_slider(0, 10, 2, "Filtration (mm Al)")

        # -----------------------
        # View mode dropdown
        # -----------------------
        self.view_selector = QComboBox()
        self.view_selector.addItems(["X-ray Projection", "Sinogram"])
        self.view_selector.currentIndexChanged.connect(self.update_projection)

        # -----------------------
        # Slide panel layout
        # -----------------------
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

        sliders.addWidget(QLabel("View Mode:"))
        sliders.addWidget(self.view_selector)

        slider_panel = QWidget()
        slider_panel.setLayout(sliders)

        # -----------------------
        # Final layout: plot + controls
        # -----------------------
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas, stretch=3)
        main_layout.addWidget(slider_panel, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # -----------------------
        # Connect slider signals
        # -----------------------
        for _, slider in [
            self.angle_slider,
            self.sid_slider,
            self.sdd_slider,
            self.kvp_slider,
            self.exp_slider,
            self.filt_slider,
        ]:
            slider.valueChanged.connect(self.update_projection)

        # -----------------------
        # Initial draw
        # -----------------------
        self.update_projection()

    # ---------------------------------------------------
    # Helper function: slider creation
    # ---------------------------------------------------
    def create_slider(self, min_val, max_val, init, label_text):
        label = QLabel(f"{label_text}: {init}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init)
        slider.label_text = label_text
        return label, slider

    # ---------------------------------------------------
    # Update projection (X-ray or Sinogram)
    # ---------------------------------------------------
    def update_projection(self):
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
        self.filt_slider[0].setText(f"Filtration (mm AL): {filt}")

        mode = self.view_selector.currentText()

        if mode == "X-ray Projection":
            img = simulate_projection_single(
                self.phantom, angle, sid, sdd, kvp, exposure, filt
            )
            title = f"X-ray Projection @ {angle}°"

        else:  # Sinogram
            img, _ = simulate_sinogram(
                self.phantom, angle, sid, sdd, kvp, exposure, filt
            )
            title = f"Sinogram (0 → {angle}°)"

        # Display
        if not hasattr(self, "im"):
            self.ax_img.clear()
            self.im = self.ax_img.imshow(
                img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto"
            )
        else:
            self.im.set_data(img)

        self.ax_img.set_title(title)

        # -----------------------
        # Intensity profile overlays (baseline + variations)
        # -----------------------
        baseline = simulate_projection(
            self.phantom,
            I0=1.0,
            sid=sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
        )

        x = np.arange(baseline.size)

        closer_sid = max(100, int(sid * 0.7))  # smaller SID -> more magnification
        dist_var = simulate_projection(
            self.phantom,
            I0=1.0,
            sid=closer_sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
        )

        dense_phantom = self.phantom * 1.25      # higher μ (denser material)
        att_var = simulate_projection(
            dense_phantom,
            I0=1.0,
            sid=sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
        )

        # Ensure angle variation is visible even at 0°
        angle_var_deg = max(5, int(angle))
        angle_var, _ = simulate_projection_angle(
            self.phantom,
            angle_var_deg,
            I0=1.0,
            sid=sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
        )

        if not hasattr(self, "profile_lines"):
            self.ax_profile.clear()
            self.profile_lines = [
                self.ax_profile.plot(x, baseline, label="Baseline", linewidth=2)[0],
                self.ax_profile.plot(x, dist_var, label=f"Closer SID {closer_sid}", linestyle="--")[0],
                self.ax_profile.plot(x, att_var, label="Higher μ (denser)", linestyle="-.")[0],
                self.ax_profile.plot(x, angle_var, label=f"Tilted {angle_var_deg}°", linestyle=":")[0],
            ]
            self.ax_profile.set_title("Intensity Profile Overlays")
            self.ax_profile.set_xlabel("Detector Position (pixels)")
            self.ax_profile.set_ylabel("Intensity")
            self.ax_profile.grid(alpha=0.2)
            self.ax_profile.legend()
            note = (
                "Notes: smaller SID spreads edges (magnification); higher μ deepens dips; "
                "tilt shifts edge positions via foreshortening."
            )
            self.profile_note = self.ax_profile.text(
                0.02, 0.95, note,
                transform=self.ax_profile.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="0.8"),
            )
        else:
            self.profile_lines[0].set_ydata(baseline)
            self.profile_lines[1].set_ydata(dist_var)
            self.profile_lines[1].set_label(f"Closer SID {closer_sid}")
            self.profile_lines[2].set_ydata(att_var)
            self.profile_lines[3].set_ydata(angle_var)
            self.profile_lines[3].set_label(f"Tilted {angle_var_deg}°")
            self.ax_profile.relim()
            self.ax_profile.autoscale_view()
            self.ax_profile.legend()

        self.canvas.draw()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    app = QApplication(sys.argv)
    gui = XrayGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
