import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ProjectFunctions.phantom import create_shepp_logan, create_breast_phantom
from ProjectFunctions.simulate_xray import (
    simulate_sinogram,
    simulate_projection_single,
    simulate_projection,
    simulate_projection_angle,
    simulate_xray_2d,
)
from ProjectFunctions.utils import roi_mean_std, roi_contrast


class XrayGUI(QMainWindow):
    def __init__(self):
        """Initialize GUI, load phantoms, build layout, and draw first view."""
        super().__init__()
        self.setWindowTitle("X-ray Simulation GUI")

        self.phantom = create_shepp_logan()
        self.breast_base, self.breast_info = create_breast_phantom()
        self.breast_compressed, self.breast_info_compressed = create_breast_phantom(
            compression=True
        )
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax_img, self.ax_profile = self.fig.subplots(2, 1)

        self.angle_slider = self.create_slider(0, 180, 30, "Angle (deg)")
        self.sid_slider   = self.create_slider(200, 1200, 500, "SID")
        self.sdd_slider   = self.create_slider(400, 1600, 1000, "SDD")
        self.kvp_slider   = self.create_slider(20, 120, 30, "kVp")
        self.exp_slider   = self.create_slider(1, 300, 100, "Exposure x0.01 s")
        self.filt_slider  = self.create_slider(0, 10, 2, "Filtration (mm Al)")

        self.view_selector = QComboBox()
        self.view_selector.addItems(["X-ray Projection", "Sinogram"])
        self.view_selector.currentIndexChanged.connect(self.update_projection)

        self.breast_toggle = QCheckBox("Use breast phantom")
        self.breast_toggle.setChecked(True)
        self.compress_toggle = QCheckBox("Compression (thinner breast)")
        self.grid_toggle = QCheckBox("Anti-scatter grid")
        for cb in [self.breast_toggle, self.compress_toggle, self.grid_toggle]:
            cb.stateChanged.connect(self.update_projection)
        self.phantom_info = QLabel("Phantom μ: adipose 0.22, gland 0.40, lesion 0.75")
        self.roi_stats = QLabel("ROI stats: N/A")

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
        sliders.addWidget(self.breast_toggle)
        sliders.addWidget(self.compress_toggle)
        sliders.addWidget(self.grid_toggle)
        sliders.addWidget(self.phantom_info)
        sliders.addWidget(self.roi_stats)

        slider_panel = QWidget()
        slider_panel.setLayout(sliders)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas, stretch=3)
        main_layout.addWidget(slider_panel, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        for _, slider in [
            self.angle_slider,
            self.sid_slider,
            self.sdd_slider,
            self.kvp_slider,
            self.exp_slider,
            self.filt_slider,
        ]:
            slider.valueChanged.connect(self.update_projection)

        self.update_projection()

    def create_slider(self, min_val, max_val, init, label_text):
        """Create horizontal slider with label; returns (label, slider)."""
        label = QLabel(f"{label_text}: {init}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init)
        slider.label_text = label_text
        return label, slider

    def update_projection(self):
        """Recompute image, profiles, ROI stats using current slider/toggle settings."""
        angle = self.angle_slider[1].value()
        sid   = self.sid_slider[1].value()
        sdd   = self.sdd_slider[1].value()
        kvp   = self.kvp_slider[1].value()
        exposure = self.exp_slider[1].value() / 100.0
        filt  = self.filt_slider[1].value()
        grid_ratio = 0.75 if self.grid_toggle.isChecked() else 1.0

        use_breast = self.breast_toggle.isChecked()
        use_compression = self.compress_toggle.isChecked() if use_breast else False
        if use_breast:
            phantom = self.breast_compressed if use_compression else self.breast_base
            phantom_info = self.breast_info_compressed if use_compression else self.breast_info
        else:
            phantom = self.phantom
            phantom_info = None
        use_external = False

        self.angle_slider[0].setText(f"Angle: {angle}°")
        self.sid_slider[0].setText(f"SID: {sid}")
        self.sdd_slider[0].setText(f"SDD: {sdd}")
        self.kvp_slider[0].setText(f"kVp: {kvp}")
        self.exp_slider[0].setText(f"Exposure x0.01s: {self.exp_slider[1].value()}")
        self.filt_slider[0].setText(f"Filtration (mm AL): {filt}")

        mode = self.view_selector.currentText()

        if mode == "X-ray Projection":
            img = simulate_xray_2d(
                phantom,
                angle,
                I0=1.0,
                sid=sid,
                sdd=sdd,
                kVp=kvp,
                exposure_time=exposure,
                filtration_mmAl=filt,
                grid_ratio=grid_ratio,
            )
            title = f"X-ray Projection @ {angle}°"

        else:
            img, _ = simulate_sinogram(
                phantom, angle, sid, sdd, kvp, exposure, filt, grid_ratio=grid_ratio
            )
            title = f"Sinogram (0 → {angle}°)"

        if not hasattr(self, "im"):
            self.ax_img.clear()
            self.im = self.ax_img.imshow(
                img, cmap="gray", vmin=0.0, vmax=0.7, aspect="auto"
            )
        else:
            self.im.set_data(img)
            self.im.set_clim(0.0, 0.7)

        self.ax_img.set_title(title)

        baseline = simulate_projection(
            phantom,
            I0=1.0,
            sid=sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
            grid_ratio=grid_ratio,
        )

        x = np.arange(baseline.size)
        need_reset_profiles = (
            not hasattr(self, "profile_lines")
            or len(self.profile_lines[0].get_xdata()) != baseline.size
        )

        closer_sid = max(100, int(sid * 0.7))
        dist_var = simulate_projection(
            phantom,
            I0=1.0,
            sid=closer_sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
            grid_ratio=grid_ratio,
        )

        dense_phantom = phantom * 1.25
        att_var = simulate_projection(
            dense_phantom,
            I0=1.0,
            sid=sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
            grid_ratio=grid_ratio,
        )

        angle_var_deg = max(5, int(angle))
        angle_var, _ = simulate_projection_angle(
            phantom,
            angle_var_deg,
            I0=1.0,
            sid=sid,
            sdd=sdd,
            kVp=kvp,
            exposure_time=exposure,
            filtration_mmAl=filt,
            grid_ratio=grid_ratio,
        )

        if use_breast:
            base_profile = simulate_projection(
                self.breast_base,
                I0=1.0,
                sid=sid,
                sdd=sdd,
                kVp=kvp,
                exposure_time=exposure,
                filtration_mmAl=filt,
                grid_ratio=grid_ratio,
            )
            compressed_profile = simulate_projection(
                self.breast_compressed,
                I0=1.0,
                sid=sid,
                sdd=sdd,
                kVp=kvp,
                exposure_time=exposure,
                filtration_mmAl=filt,
                grid_ratio=grid_ratio,
            )
        else:
            base_profile = baseline
            compressed_profile = baseline

        def _match_length(arr, target_len):
            if arr.size == target_len:
                return arr
            xp = np.linspace(0, 1, arr.size)
            xq = np.linspace(0, 1, target_len)
            return np.interp(xq, xp, arr)

        dist_var = _match_length(dist_var, baseline.size)
        att_var = _match_length(att_var, baseline.size)
        angle_var = _match_length(angle_var, baseline.size)
        base_profile = _match_length(base_profile, baseline.size)
        compressed_profile = _match_length(compressed_profile, baseline.size)
        x = np.arange(baseline.size)
        need_reset_profiles = (
            not hasattr(self, "profile_lines")
            or len(self.profile_lines[0].get_xdata()) != baseline.size
        )

        if need_reset_profiles:
            self.ax_profile.clear()
            self.profile_lines = [
                self.ax_profile.plot(x, baseline, label="Baseline", linewidth=2)[0],
                self.ax_profile.plot(x, dist_var, label=f"Closer SID {closer_sid}", linestyle="--")[0],
                self.ax_profile.plot(x, att_var, label="Higher μ (denser)", linestyle="-.")[0],
                self.ax_profile.plot(x, angle_var, label=f"Tilted {angle_var_deg}°", linestyle=":")[0],
                self.ax_profile.plot(x, compressed_profile, label="Compressed phantom", linestyle="-.", color="tab:red")[0],
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
            self.profile_lines[4].set_ydata(compressed_profile)
            self.ax_profile.relim()
            self.ax_profile.autoscale_view()
            self.ax_profile.legend()

        if use_breast:
            lesion_mean, lesion_std = roi_mean_std(
                phantom, phantom_info["lesion_mask"]
            )
            bg_mean, bg_std = roi_mean_std(
                phantom, phantom_info["background_mask"]
            )
            contrast = roi_contrast(lesion_mean, bg_mean)

            base_lesion, _ = roi_mean_std(self.breast_base, self.breast_info["lesion_mask"])
            base_bg, _ = roi_mean_std(self.breast_base, self.breast_info["background_mask"])
            comp_lesion, _ = roi_mean_std(self.breast_compressed, self.breast_info_compressed["lesion_mask"])
            comp_bg, _ = roi_mean_std(self.breast_compressed, self.breast_info_compressed["background_mask"])
            comp_contrast = roi_contrast(comp_lesion, comp_bg)
            base_contrast = roi_contrast(base_lesion, base_bg)

            self.roi_stats.setText(
                f"Current ROI μ: lesion {lesion_mean:.3f}±{lesion_std:.3f}, "
                f"bg {bg_mean:.3f}±{bg_std:.3f}, contrast {contrast:.2f}\n"
                f"Baseline vs compressed contrast: {base_contrast:.2f} → {comp_contrast:.2f}"
            )
            self.phantom_info.setText(
                "Phantom μ: adipose 0.22, gland 0.40, lesion 0.75"
            )
        else:
            self.roi_stats.setText("ROI stats: N/A (toggle breast phantom)")
            self.phantom_info.setText("Phantom: Shepp-Logan (no labeled ROIs)")

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    gui = XrayGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
