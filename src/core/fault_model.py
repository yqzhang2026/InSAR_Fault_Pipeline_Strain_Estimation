from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from shapely.geometry import LineString, Point, Polygon

from src.config import AppConfig


@dataclass
class ModelFitResult:
    params: np.ndarray
    y_pred: np.ndarray
    rmse: float
    r2: float
    pcov: Optional[np.ndarray] = None
    param_std: Optional[np.ndarray] = None


def sigmoid(x, a, b, c, d):
    return a * expit(b * (x - c)) + d


def _fit_sigmoid(x: np.ndarray, y: np.ndarray, cfg: AppConfig) -> ModelFitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(min(x.size, y.size))
    if n <= 0:
        raise ValueError("Sigmoid fitting requires non-empty samples.")
    x = x[:n]
    y = y[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 4:
        raise ValueError("Sigmoid fitting requires at least 4 finite samples.")
    # Use data-driven initialization by default so parameters come from the run itself.
    a0 = float(np.nanmax(y) - np.nanmin(y))
    d0 = float(np.nanmin(y))
    x_std = max(float(np.nanstd(x)), 1e-6)
    y_std = max(float(np.nanstd(y)), 1e-6)
    b0 = float(np.clip(y_std / x_std, 1e-3, 2.0))
    p0 = [
        a0 if np.isfinite(a0) else cfg.sigmoid_a,
        b0,
        float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else float(getattr(cfg, "sigmoid_c", 0.0)),
        d0 if np.isfinite(d0) else cfg.sigmoid_d,
    ]
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
    lower = [-5.0 * y_span, 1e-8, float(np.nanmin(x) - x_span), float(np.nanmin(y) - 2.0 * y_span)]
    upper = [5.0 * y_span, 10.0, float(np.nanmax(x) + x_span), float(np.nanmax(y) + 2.0 * y_span)]

    def _curve_fit_with_bounds(p0_fit, b_lower):
        lower_fit = [lower[0], float(b_lower), lower[2], lower[3]]
        return curve_fit(sigmoid, x, y, p0=p0_fit, bounds=(lower_fit, upper), maxfev=3000000)

    def _robust_amp(arr: np.ndarray) -> float:
        v = np.asarray(arr, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0
        q95 = float(np.nanpercentile(v, 95.0))
        q05 = float(np.nanpercentile(v, 5.0))
        return max(q95 - q05, 0.0)
    try:
        popt, pcov = _curve_fit_with_bounds(p0, b_lower=1e-8)
    except Exception:
        p0_retry = [
            a0 if np.isfinite(a0) else cfg.sigmoid_a,
            max(b0, 0.05),
            float(getattr(cfg, "sigmoid_c", 0.0)),
            d0 if np.isfinite(d0) else cfg.sigmoid_d,
        ]
        popt, pcov = _curve_fit_with_bounds(p0_retry, b_lower=1e-8)
    y_pred = sigmoid(x, *popt)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    obs_amp = _robust_amp(y)
    pred_amp = _robust_amp(y_pred)
    degenerate_flat = bool(obs_amp > 1e-3 and pred_amp < 0.12 * obs_amp)
    if degenerate_flat:
        p0_hist = [
            a0 if np.isfinite(a0) else cfg.sigmoid_a,
            0.2 if np.nanstd(x) > 1e-9 else 0.01,
            float(getattr(cfg, "sigmoid_c", 0.0)),
            d0 if np.isfinite(d0) else cfg.sigmoid_d,
        ]
        try:
            popt_hist, pcov_hist = _curve_fit_with_bounds(p0_hist, b_lower=1e-4)
            y_pred_hist = sigmoid(x, *popt_hist)
            rmse_hist = float(np.sqrt(np.mean((y - y_pred_hist) ** 2)))
            pred_amp_hist = _robust_amp(y_pred_hist)
            improves_shape = pred_amp_hist > pred_amp * 1.5
            acceptable_rmse = rmse_hist <= rmse * 1.02
            if (improves_shape and acceptable_rmse) or (pred_amp_hist >= 0.4 * obs_amp and rmse_hist <= rmse * 1.05):
                popt = popt_hist
                pcov = pcov_hist
                y_pred = y_pred_hist
                rmse = rmse_hist
        except Exception:
            pass
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pcov = np.asarray(pcov, dtype=float)
    if pcov.shape != (4, 4) or not np.isfinite(pcov).all():
        pcov = None
        param_std = None
    else:
        diag = np.clip(np.diag(pcov), 0.0, np.inf)
        param_std = np.sqrt(diag)
    params_arr = np.asarray(popt, dtype=float)
    return ModelFitResult(
        params=params_arr,
        y_pred=y_pred,
        rmse=rmse,
        r2=r2,
        pcov=pcov,
        param_std=param_std,
    )


def _fit_piecewise_linear(x: np.ndarray, y: np.ndarray) -> ModelFitResult:
    left_mask = x < 0
    right_mask = ~left_mask
    if left_mask.sum() < 3 or right_mask.sum() < 3:
        # fallback to global linear if one side too sparse
        p = np.polyfit(x, y, 1)
        y_pred = np.polyval(p, x)
        params = np.array([p[0], p[1], p[0], p[1]], dtype=float)
    else:
        pl = np.polyfit(x[left_mask], y[left_mask], 1)
        pr = np.polyfit(x[right_mask], y[right_mask], 1)
        y_pred = np.where(x < 0, np.polyval(pl, x), np.polyval(pr, x))
        params = np.array([pl[0], pl[1], pr[0], pr[1]], dtype=float)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return ModelFitResult(params=params, y_pred=y_pred, rmse=rmse, r2=r2)


def fit_model(config: AppConfig, signed_distance: np.ndarray, displacement_xy: np.ndarray) -> Dict:
    x = np.asarray(signed_distance).reshape(-1)
    yx = np.asarray(displacement_xy)[:, 0]
    yy = np.asarray(displacement_xy)[:, 1]

    if config.model_type == "sigmoid":
        fit_x = _fit_sigmoid(x, yx, config)
        fit_y = _fit_sigmoid(x, yy, config)
    elif config.model_type == "piecewise_linear":
        fit_x = _fit_piecewise_linear(x, yx)
        fit_y = _fit_piecewise_linear(x, yy)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    return {
        "x_component": fit_x,
        "y_component": fit_y,
        "model_type": config.model_type,
    }


def _build_side_polygons(fault_coords: np.ndarray, points: np.ndarray, margin: float) -> Tuple[Polygon, Polygon]:
    xmin = float(points[:, 0].min() - margin)
    xmax = float(points[:, 0].max() + margin)
    ymin = float(points[:, 1].min() - margin)
    ymax = float(points[:, 1].max() + margin)

    left_bottom = np.array([[xmin, ymin]])
    left_top = np.array([[xmin, ymax]])
    right_top = np.array([[xmax, ymax]])
    right_bottom = np.array([[xmax, ymin]])

    left_area_line = np.concatenate([fault_coords, left_bottom, left_top])
    right_area_line = np.concatenate([fault_coords, right_top, right_bottom])
    return Polygon(left_area_line), Polygon(right_area_line)


def compute_signed_distance_to_polyline(
    points: np.ndarray, polyline_coords: np.ndarray, flip_sign: bool = False
) -> np.ndarray:
    """
    Signed distance using nearest segment projection.
    Sign convention: right side is positive, left side is negative
    with respect to segment direction (polyline vertex order).
    """
    pts = np.asarray(points, dtype=float)
    line = np.asarray(polyline_coords, dtype=float)
    if line.ndim != 2 or line.shape[0] < 2 or line.shape[1] != 2:
        raise ValueError("polyline_coords must have shape (N, 2), N >= 2")

    seg_start = line[:-1]
    seg_end = line[1:]
    seg_vec = seg_end - seg_start
    seg_len2 = np.einsum("ij,ij->i", seg_vec, seg_vec)
    seg_len2 = np.where(seg_len2 <= 1e-12, 1e-12, seg_len2)

    signed = np.zeros(len(pts), dtype=float)
    chunk = 2048
    for beg in range(0, len(pts), chunk):
        end = min(len(pts), beg + chunk)
        pblk = pts[beg:end]
        d = pblk[:, None, :] - seg_start[None, :, :]
        t = np.einsum("bij,ij->bi", d, seg_vec) / seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
        rel = pblk[:, None, :] - proj
        d2 = np.einsum("bij,bij->bi", rel, rel)
        j = np.argmin(d2, axis=1)
        row = np.arange(end - beg)
        rel_best = rel[row, j]
        seg_best = seg_vec[j]
        dist = np.sqrt(np.maximum(d2[row, j], 0.0))
        cross = seg_best[:, 0] * rel_best[:, 1] - seg_best[:, 1] * rel_best[:, 0]
        sign = np.where(np.abs(cross) <= 1e-12, 1.0, np.where(cross > 0.0, -1.0, 1.0))
        signed[beg:end] = sign * dist
    if flip_sign:
        signed = -signed
    return signed


def compute_fault_distance_field(config: AppConfig, points: np.ndarray, fault: LineString) -> Dict:
    fault_simplified = fault.simplify(config.fault_simplify_tolerance, preserve_topology=False)
    fault_coords = np.asarray(fault_simplified.coords, dtype=float)
    signed = compute_signed_distance_to_polyline(
        points=np.asarray(points, dtype=float),
        polyline_coords=fault_coords,
        flip_sign=bool(getattr(config, "flip_signed_distance", False)),
    )

    return {
        "signed_distance": np.asarray(signed, dtype=float),
        "fault_simplified": fault_simplified,
    }
