# Figure 2: minimal reproducible seed-42 demonstration

This repository contains a reproducible numerical examination of the points in figure 2 of:

Hang Yuan (2025), *Artificial intelligence in language learning: biometric feedback and adaptive reading for improved comprehension and reduced anxiety*, Humanities and Social Sciences Communications 12:556, Figure 2, PDF p.9. https://doi.org/10.1057/s41599-025-04878-w

This is a summary of a broader audit conducted with the help of an AI assistant. The paper contains also a number of other inconsistencies that are not discussed in this repository. The aim of this repository is just to demonstrate the compatibility of the data plotted in fig.2 with a synthetic data generation procedure.

## Files

- `figure2_original.jpeg`: original **1950 × 1296** embedded image, extracted without rendering/resizing; identical to the supplied JPEG and PDF image object 364. SHA-256: `9db0e15eb22866bbf9f16d21f3fbb98306bb7ad001f41457ff668d33830a6a1a`.
- `extract_points.py`: image-only cross detection, tick calibration and coordinate export. 
- `figure2_seed42.ipynb`: six executable cells, explanatory text, the exact candidate generator, paired 2D matching, a correspondence/residual plot, and residual histograms in axis units. Saved with executed results and embedded images.
- `requirements.txt`: tested package versions for Python 3.10.
- `outputs/`: reproducible CSVs, calibration and images; included for immediate inspection. Re-running replaces these derived files.

## Run

Use Python **3.10** for the pinned package versions:

```bash
cd code_shared
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python extract_points.py
```

Open `figure2_seed42.ipynb` in Jupyter or VS Code, select this environment, and **Run All**. Use this folder as the notebook working directory. The first cell reruns extraction automatically; running the separate script first is optional. 

To execute and save all notebook outputs without an interface:

```bash
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('figure2_seed42.ipynb').resolve()
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(path.parent)}}).execute()
nbformat.write(notebook, path)
PY
```


## Expected results

| Group | Image-only detections | Candidate points | Paired matches | RMS pixel discrepancy | Maximum pixel discrepancy |
|---|---:|---:|---:|---:|---:|
| Blue (experimental) | 100 | 100 | 95 | 0.243 | 0.724 |
| Red (control) | 96 | 100 | 95 | 0.372 | 0.746 |

These 190 matches persist at 1-, 2-, 3- and 5-pixel matching thresholds. Six detected peaks and ten generated positions remain unmatched and are explicitly plotted/exported. Small software-version changes can affect tied local peaks; the image-processing versions are pinned for reproducibility.

The legacy generator is:

```python
rng = np.random.RandomState(42)
x_blue = rng.normal(220, 15, 100)
y_blue = rng.normal(4, 1, 100)
u_blue = rng.uniform(0, 1, 100)
x_red = rng.normal(280, 20, 100)
y_red = rng.normal(7, 1.5, 100)
u_red = rng.uniform(0, 1, 100)
```

The intervening uniform draws are essential for reproducing the red coordinates. `np.random.seed(42)` followed by equivalent legacy calls gives the same sequence; `np.random.default_rng(42)` does not. Uniform values are retained as candidate marker-size draws; this minimal demonstration does not estimate the absolute saccade-size scale or test marker-size correlations.

## Coordinates and extraction parameters

`outputs/recovered_points.csv` contains **all 196 image-only detections**, preserving x/y pairing:

- `group`: blue/red; `point_id`: detector identifier, **not a participant ID**.
- `x_pixel`, `y_pixel`: subpixel centers in the original image; origin upper-left, x right, y down.
- `fixation_ms`, `regression_count`: coordinates in the units labelled on the axes. The figure says “Regression Counts”; the paper's table describes regressions per 100 words.
- `center_uncertainty_px`, `fixation_uncertainty_ms`, `regression_uncertainty`: conservative working allowances, **not confidence intervals**.
- `detection_score`: evidence for both diagonal arms.

The script localizes seven labelled x ticks (200–320, step 20) and eight y ticks (2–9), starting from fixed manually identified grid-line neighborhoods. It fits:

```
x_pixel ≈ 11.000402531 × fixation_ms − 1831.927866139
y_pixel ≈ −109.941945021 × regression_count + 1384.524313136
```

Exact tick positions, slopes/intercepts and image hash are saved in `outputs/calibration.json`. This calibration and the plot bounds are **specific to the included original image**, not a general-purpose plot digitizer.

Detector: color-excess threshold 35; two blurred diagonal kernels of radius 9; minimum arm response; local-peak separation 9 pixels; score threshold .45 blue/.30 red; subpixel diagonal-ridge refinement in a 13 × 13 neighborhood. Working center uncertainty is 1.5 pixels for isolated points (nearest center >42 pixels) and 3 pixels otherwise, with another .5 pixel allowed for calibration in the converted-unit fields. All parameters are visible in the script.

## Matching and outputs

The notebook projects generated axis coordinates into the **exact calibrated image-pixel system** and uses squared-distance Hungarian assignment with dummy unmatched slots costing 3² per detection. This differs negligibly from the earlier audit's rounded 11/110 pixel scales. Each point can be paired at most once. Original x/y pairing is retained, and no sorted-marginal fit or transformation is estimated here.

- `recovered_points.csv`, `calibration.json`, `detected_points.png`: produced by the script.
- `simulated_points.csv`: all 200 candidate points in axis and pixel units, with RNG indices and uniform draws.
- `matched_pairs.csv`: the 190 paired IDs and residuals in both coordinate systems. Residual = recovered − simulated; pixel-y points down, regression units up.
- `unmatched_detections.csv`, `unmatched_predictions.csv`: every unmatched point, without silent omission.
- `matching_summary.csv`: counts and errors for assigned pairs only.
- `correspondence.png`: all points with unmatched markers distinguished, and residuals zoomed to ±1 pixel.
- `residuals_in_axis_units.png`: matched fixation/regression residual histograms.

## Interpretation

This is a reproduction of an exploratorily identified candidate. It deliberately excludes the full audit's seed search. The close correspondence concerns **paired 2D locations**, not merely similar Gaussian distributions. It supports a specific synthetic procedure compatible with the published figure.
