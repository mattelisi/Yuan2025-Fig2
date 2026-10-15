#!/usr/bin/env python3
"""Digitise Figure 2 using image evidence only (no RNG values or model-assisted edits).

Usage: python extract_points.py [--image figure2_original.jpeg] [--outdir outputs]
Pixel origin: upper-left; x increases right, y increases down. The hard-coded
labelled tick neighborhoods and plot bounds are specific to the supplied JPEG.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

HERE = Path(__file__).resolve().parent


def color_excess(image, color):
    """Separate blue/red ink from gray axes, grid and background (OpenCV BGR)."""
    b, g, r = image.astype(float).transpose(2, 0, 1)
    return np.maximum(0, b - np.maximum(r, g) if color == 'blue'
                      else r - np.maximum(b, g))


def calibrate(image):
    """Fit pixel = slope * axis_value + intercept from labelled grid lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_ticks, y_ticks = [], []
    # Labels were read directly from the original figure; positions are localized
    # within +/-4 pixels of the manually identified grid-line neighborhoods.
    for value, guess in zip(range(200, 321, 20),
                            [368, 588, 808, 1028, 1248, 1468, 1688]):
        positions = np.arange(guess - 4, guess + 5)
        weights = (255 - gray[300:1180, guess - 4:guess + 5]).mean(axis=0)
        x_ticks.append([value, float(np.average(positions, weights=weights))])
    for value, guess in zip(range(2, 10),
                            [1165, 1055, 945, 835, 725, 615, 505, 395]):
        positions = np.arange(guess - 4, guess + 5)
        weights = (255 - gray[guess - 4:guess + 5, 100:1900]).mean(axis=1)
        y_ticks.append([value, float(np.average(positions, weights=weights))])
    ax, bx = np.polyfit(*np.array(x_ticks).T, 1)
    ay, by = np.polyfit(*np.array(y_ticks).T, 1)
    return dict(x_ticks_value_pixel=x_ticks, y_ticks_value_pixel=y_ticks,
                pixel_x_per_ms=ax, pixel_x_intercept=bx,
                pixel_y_per_regression=ay, pixel_y_intercept=by,
                x_tick_residual_sd_px=float(np.std(np.array(x_ticks)[:, 1] -
                    (ax * np.array(x_ticks)[:, 0] + bx))),
                y_tick_residual_sd_px=float(np.std(np.array(y_ticks)[:, 1] -
                    (ay * np.array(y_ticks)[:, 0] + by))))


def detect_centers(excess, threshold):
    """Require both diagonal arms; suppress nearby peaks, then refine centers."""
    ink = excess.copy()
    ink[:70] = 0
    ink[1200:] = 0
    ink[:, :90] = 0
    ink[:, 1900:] = 0
    mask = (ink > 35).astype('float32')
    responses = []
    radius = 9
    for sign in [-1, 1]:
        kernel = np.zeros((19, 19), np.float32)
        for j in range(-radius, radius + 1):
            kernel[radius + j, radius + sign * j] = 1 / 19
        kernel = cv2.GaussianBlur(kernel, (3, 3), 0.6)
        responses.append(cv2.filter2D(mask, -1, kernel))
    score = np.minimum(*responses)
    peaks = peak_local_max(score, min_distance=9, threshold_abs=threshold)
    centers = []
    for y, x in peaks:
        yy, xx = np.mgrid[y - 6:y + 7, x - 6:x + 7]
        weights = excess[y - 6:y + 7, x - 6:x + 7]
        offsets = []
        for sign in [-1, 1]:
            distance = (yy - y) - sign * (xx - x)
            selected = abs(distance) < 3
            offsets.append(np.sum(distance[selected] * weights[selected]) /
                           max(1, weights[selected].sum()))
        centers.append([x + (offsets[0] - offsets[1]) / 2,
                        y + (offsets[0] + offsets[1]) / 2])
    return np.asarray(centers), score[peaks[:, 0], peaks[:, 1]]


def extract(image_path=HERE / 'figure2_original.jpeg', outdir=HERE / 'outputs'):
    image_path, outdir = Path(image_path), Path(outdir)
    image = cv2.imread(str(image_path))
    if image is None or image.shape != (1296, 1950, 3):
        raise ValueError('Expected the original 1950 x 1296 RGB Figure 2 JPEG.')
    outdir.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(image)
    calibration['image_sha256'] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    calibration['detector'] = dict(chroma_cutoff=35, radius=9, min_distance=9,
                                  blue_threshold=0.45, red_threshold=0.30)
    ax, bx = calibration['pixel_x_per_ms'], calibration['pixel_x_intercept']
    ay, by = calibration['pixel_y_per_regression'], calibration['pixel_y_intercept']
    records = []
    annotated = image.copy()
    for color, threshold in [('blue', 0.45), ('red', 0.30)]:
        centers, scores = detect_centers(color_excess(image, color), threshold)
        distances = cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)
        for index, ((x, y), score) in enumerate(zip(centers, scores), start=1):
            # Conservative allowances, not confidence intervals. An additional
            # 0.5 pixel allows for axis calibration when converting to data units.
            uncertainty = 1.5 if distances[index - 1].min() > 42 else 3.0
            records.append(dict(group=color, point_id=index, x_pixel=x, y_pixel=y,
                fixation_ms=(x - bx) / ax, regression_count=(y - by) / ay,
                center_uncertainty_px=uncertainty,
                fixation_uncertainty_ms=(uncertainty + 0.5) / ax,
                regression_uncertainty=(uncertainty + 0.5) / abs(ay),
                detection_score=float(score)))
            cv2.circle(annotated, (round(x), round(y)), 6, (0, 120, 0), 1)
            cv2.putText(annotated, str(index), (round(x) + 5, round(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 70, 0), 1)
    points = pd.DataFrame(records)
    points.to_csv(outdir / 'recovered_points.csv', index=False)
    (outdir / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    cv2.imwrite(str(outdir / 'detected_points.png'), annotated)
    return points, calibration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=HERE / 'figure2_original.jpeg')
    parser.add_argument('--outdir', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    points, calibration = extract(args.image, args.outdir)
    print(points.groupby('group').size().rename('detected_centers').to_string())
    print(f'Coordinates, calibration and annotation saved in {args.outdir.resolve()}')
