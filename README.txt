## Code Execution Instructions

### Environment
- Python 3.10+ recommended.
- Create and activate a virtual environment:
  ```
  python -m venv .venv
  source .venv/bin/activate   # on Windows: .venv\Scripts\activate
  ```

### Install prerequisites
- From the repo root (`Xray/`), install required packages:
  ```
  pip install -r requirements.txt
  ```
Packages used: numpy, matplotlib, scikit-image, PyQt5.

### Datasets / external resources
- None required. All phantoms are generated procedurally; no downloads needed.

### Run the GUI (generate interactive results)
```
python gui.py
```
Suggested starting sliders: angle 0–10°, SID 500, SDD 1000, kVp 30–40, exposure x0.01s 80–120, filtration 2–3 mm Al, grid off for brightness. Compression toggle compares thickness. The GUI shows the radiograph/sinogram and profile overlays; ROI stats/contrast are displayed in the sidebar.

### Regenerate figures (non-interactive)
```
python generate_figures.py
```
Outputs saved under `figs/` and prints ROI/contrast stats for baseline and compressed phantoms. No training/testing steps apply (not an AI model).

## Generated figures (in `figs/`)
- `baseline_radiograph.png`: baseline breast radiograph (0°, SID 500/SDD 1000, kVp 35, exposure 1s, 2mm Al).
- `distance_variation.png`: baseline vs closer (SID 350) vs farther (SID 700) showing magnification/edge shifts.
- `mu_variation.png`: baseline vs denser (higher μ) material.
- `angle_variation.png`: radiographs at 0/15/30 degrees.
- `profile_overlays.png`: baseline profile with overlays for distance, μ, angle.
- `profile_compressed.png`: baseline vs compressed breast profile.
- `sinogram.png`: 0–180° sinogram.
- `phantom_schematic.png`: μ map schematic with skin rim, pectoral wedge, gland core, lesion, calc spots.

## How this maps to the project proposal
- GUI controls for beam energy (kVp), SID/SDD, angle, exposure, filtration, grid: **present** (`gui.py`).
- Breast phantom with lesion, calc spots, benign mass, compression toggle: **present** (`phantom.py`, used in GUI and figures).
- 2D radiograph and sinogram views: **present** (`gui.py`, `simulate_xray.py`).
- Intensity profiles baseline + distance/μ/angle overlays with notes: **present** (`gui.py` bottom plot, `figs/profile_overlays.png`).
- Compressed vs baseline comparison: **present** (GUI compression toggle; `figs/profile_compressed.png`).
- ROI stats/contrast for lesion vs background: **present** (GUI sidebar text; printed in `generate_figures.py` output).
- Schematic/diagram with attenuation labels: **present** (`figs/phantom_schematic.png`).
- Saved figures for baseline and parameter variations: **present** (`figs/`).
- Baseline 2D projection of “3D phantom”: **not implemented** (we use 2D phantom); can add a simple 3D-to-2D projection if required.
- Quantitative tables for parameter variations: basic ROI/contrast numbers printed by `generate_figures.py`; extend to CSV if needed.
