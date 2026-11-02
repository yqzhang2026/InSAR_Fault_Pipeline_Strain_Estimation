from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.core.pipeline_strain import _resolve_fit_params_with_override, predict_pipe_motion_and_strain

YEAR_DAYS = 365.25


def _emit(progress: Optional[Callable[[int, str], None]], pct: int, msg: str) -> None:
    if progress:
        progress(pct, msg)


def _fiber_xy_from_array(arr: np.ndarray, *, spacing_m: float, origin_offset_m: float) -> tuple[np.ndarray, np.ndarray, bool]:
    data = np.asarray(arr)
    if data.dtype == object:
        raise ValueError("Fiber npy must be a numeric array, not object/pickle data.")
    data = np.squeeze(data)
    x_from_file = False
    if data.ndim == 0:
        raise ValueError("Fiber npy is a scalar.")
    if data.ndim == 1:
        y = data.astype(float, copy=False)
    elif data.ndim == 2:
        if data.shape[0] == 2 and data.shape[1] != 2:
            data = data.T
        if data.shape[1] == 1:
            y = data[:, 0].astype(float, copy=False)
        elif data.shape[1] >= 2:
            x_from_file = True
            x = data[:, 0].astype(float, copy=False)
            y = data[:, -1].astype(float, copy=False)
            y = np.asarray(y, dtype=float).reshape(-1)
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != y.size:
                n = min(x.size, y.size)
                x, y = x[:n], y[:n]
            if y.size < 2:
                raise ValueError("Fiber array does not contain enough numeric values.")
            return x, y, True
        else:
            raise ValueError(f"Unsupported fiber npy shape: {tuple(data.shape)}")
    else:
        raise ValueError(f"Unsupported fiber npy ndim={data.ndim}, shape={tuple(data.shape)}")
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 2:
        raise ValueError("Fiber array does not contain enough numeric values.")
    x = float(origin_offset_m) + np.arange(int(y.size), dtype=float) * float(spacing_m)
    return x, y, x_from_file


def load_fiber_yearly_profile(
    path: str | Path,
    *,
    spacing_m: float,
    origin_offset_m: float = -119.0,
    reverse: bool = True,
    skip_header: bool = True,
    fiber_is_microstrain: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Fiber file not found: {fp}")
    ext = fp.suffix.lower()
    x_from_file = False
    if ext == ".npy":
        arr = np.load(fp, allow_pickle=False)
        x, y, x_from_file = _fiber_xy_from_array(arr, spacing_m=spacing_m, origin_offset_m=origin_offset_m)
    else:
        df = pd.read_csv(fp, header=0 if skip_header else None)
        if df.shape[1] < 1 or df.shape[0] < 2:
            raise ValueError(f"Fiber CSV has too few rows/columns: {fp}")
        y = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).sum() < 2:
            raise ValueError("Fiber CSV last column does not contain enough numeric values.")
        x = float(origin_offset_m) + np.arange(int(y.size), dtype=float) * float(spacing_m)
    if reverse:
        y = y[::-1].copy()
    if not fiber_is_microstrain:
        y = y * 1e6
    return x, y, x_from_file


def yearly_axial_microstrain(axial_strain: np.ndarray, day_diff: float) -> np.ndarray:
    dd = float(day_diff)
    if not np.isfinite(dd) or abs(dd) < 1e-12:
        raise ValueError("Selected date has zero day_diffs; choose a later date.")
    return np.asarray(axial_strain, dtype=float) * 1e6 * YEAR_DAYS / dd


def rmse_on_overlap(model_x: np.ndarray, model_y: np.ndarray, fiber_x: np.ndarray, fiber_y: np.ndarray) -> float:
    mx = np.asarray(model_x, dtype=float).reshape(-1)
    my = np.asarray(model_y, dtype=float).reshape(-1)
    fx = np.asarray(fiber_x, dtype=float).reshape(-1)
    fy = np.asarray(fiber_y, dtype=float).reshape(-1)
    m_ok = np.isfinite(mx) & np.isfinite(my)
    f_ok = np.isfinite(fx) & np.isfinite(fy)
    mx, my = mx[m_ok], my[m_ok]
    fx, fy = fx[f_ok], fy[f_ok]
    if mx.size < 2 or fx.size < 2:
        raise ValueError("Need at least 2 finite points on both fiber and model profiles.")
    order = np.argsort(fx)
    fx, fy = fx[order], fy[order]
    xmin = max(float(np.nanmin(mx)), float(np.nanmin(fx)))
    xmax = min(float(np.nanmax(mx)), float(np.nanmax(fx)))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        raise ValueError("Fiber and model x ranges do not overlap. Check spacing/origin/reverse.")
    mask = (mx >= xmin) & (mx <= xmax)
    if int(mask.sum()) < 2:
        raise ValueError("Not enough overlapping x points for RMSE.")
    xq = mx[mask]
    yq = my[mask]
    y_fiber = np.interp(xq, fx, fy)
    diff = yq - y_fiber
    return float(np.sqrt(np.mean(diff * diff)))


def peak_x(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    n = min(xx.size, yy.size)
    xx, yy = xx[:n], yy[:n]
    ok = np.isfinite(xx) & np.isfinite(yy)
    if int(ok.sum()) < 1:
        raise ValueError("No finite points to locate |strain| peak.")
    local = np.argmax(np.abs(yy[ok]))
    idx = int(np.flatnonzero(ok)[int(local)])
    return float(xx[idx])


def origin_from_peaks(
    fiber_x: np.ndarray,
    fiber_y: np.ndarray,
    model_x: np.ndarray,
    model_y: np.ndarray,
) -> float:
    return peak_x(model_x, model_y) - peak_x(fiber_x, fiber_y)


def _fiber_x_aligned(
    x_base: np.ndarray,
    fiber_y: np.ndarray,
    model_x: np.ndarray,
    model_y: np.ndarray,
    *,
    auto_origin: bool,
    origin_offset_m: float,
    x_from_file: bool,
) -> tuple[np.ndarray, float]:
    x0 = np.asarray(x_base, dtype=float).reshape(-1)
    if auto_origin:
        origin = float(origin_from_peaks(x0, fiber_y, model_x, model_y))
        return x0 + origin, origin
    if x_from_file:
        return x0, float(origin_offset_m)
    return x0 + float(origin_offset_m), float(origin_offset_m)


def _b_grid_from_step(b_min: float, b_max: float, b_step: float) -> np.ndarray:
    lo = float(b_min)
    hi = float(b_max)
    step = float(b_step)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("b_min and b_max must be finite.")
    if hi < lo:
        lo, hi = hi, lo
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be > 0.")
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    if n < 1:
        return np.array([lo], dtype=float)
    grid = lo + np.arange(n, dtype=float) * step
    if grid[-1] < hi - 1e-9:
        grid = np.append(grid, hi)
    return grid


def _current_fault_parallel_params(state: Dict, epoch_idx: int) -> np.ndarray:
    fit_t = (state.get("tangential_fit_by_epoch") or {}).get(epoch_idx)
    if fit_t is None:
        raise ValueError("Fault-parallel fit is missing for the selected date. Run Step 2 first.")
    overrides = state.get("fault_fit_override_by_epoch", {})
    epoch_override = overrides.get(epoch_idx, {}) if isinstance(overrides, dict) else {}
    custom = epoch_override.get("Fault-parallel") if isinstance(epoch_override, dict) else None
    return _resolve_fit_params_with_override(fit_t, custom)


def _overrides_with_fault_parallel_b(state: Dict, epoch_idx: int, b_value: float) -> Dict[int, Dict[str, np.ndarray]]:
    existing = copy.deepcopy(state.get("fault_fit_override_by_epoch", {}))
    if not isinstance(existing, dict):
        existing = {}
    params = _current_fault_parallel_params(state, epoch_idx).astype(float, copy=True)
    params[1] = float(b_value)
    epoch_override = dict(existing.get(epoch_idx, {})) if isinstance(existing.get(epoch_idx, {}), dict) else {}
    epoch_override["Fault-parallel"] = params
    existing[epoch_idx] = epoch_override
    return existing


def _predict_yearly_profile(config: AppConfig, state: Dict, epoch_idx: int, fit_overrides: Dict) -> tuple[np.ndarray, np.ndarray]:
    nez = np.asarray(state["NEZ"])
    pipe = predict_pipe_motion_and_strain(
        config=config,
        pipe_axis=state["data"]["pipe_axis"],
        pipe_axis_segments=state["data"].get("pipe_axis_segments"),
        fault_line=state["notebook_fault_ref"]["fault_line_truth"],
        left_area=state["notebook_fault_ref"]["left_area"],
        fault_polyline_for_projection=np.asarray(state["notebook_fault_ref"]["fl2"]),
        models_by_epoch=state["models_by_epoch"],
        tangential_fit_by_epoch=state.get("tangential_fit_by_epoch", {}),
        normal_fit_by_epoch=state.get("normal_fit_by_epoch", {}),
        fit_overrides_by_epoch=fit_overrides,
        pair_count=int(nez.shape[0]),
        intersection_point=state["fit_band"]["intersection"],
        compute_strain=True,
    )
    axial = np.asarray(pipe.get("axial_strain_smoothed", []), dtype=float)
    if axial.ndim != 2 or epoch_idx >= axial.shape[0]:
        raise ValueError("Kernel did not return axial strain for the selected date.")
    matched = state.get("matched", {}) if isinstance(state.get("matched", {}), dict) else {}
    day_diffs = np.asarray(matched.get("day_diffs", []), dtype=float).reshape(-1)
    if epoch_idx >= day_diffs.size:
        raise ValueError("day_diffs is missing for the selected date.")
    model_x = np.asarray(pipe.get("s_nodes_offset", []), dtype=float).reshape(-1)
    model_y = yearly_axial_microstrain(axial[epoch_idx], float(day_diffs[epoch_idx]))
    if model_x.size != model_y.size:
        n = min(model_x.size, model_y.size)
        model_x = model_x[:n]
        model_y = model_y[:n]
    return model_x, model_y


def run_fiber_b_calibration(
    config: AppConfig,
    state: Dict,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    required = ["models_by_epoch", "NEZ", "data", "matched", "tangential_fit_by_epoch", "notebook_fault_ref", "fit_band"]
    if any(k not in state for k in required):
        raise ValueError("Fiber calibration requires Step 2 results. Run Step 2 first.")
    req = state.get("fiber_calibration_request", {}) if isinstance(state.get("fiber_calibration_request", {}), dict) else {}

    epoch_idx = int(req.get("epoch_idx", 0))
    matched = state.get("matched", {})
    pairs = matched.get("matched_pairs", []) if isinstance(matched, dict) else []
    if not pairs:
        raise ValueError("No matched dates. Run Step 1–2 first.")
    epoch_idx = min(max(epoch_idx, 0), len(pairs) - 1)
    day_diffs = np.asarray(matched.get("day_diffs", []), dtype=float).reshape(-1)
    if epoch_idx >= day_diffs.size or abs(float(day_diffs[epoch_idx])) < 1e-12:
        raise ValueError("Selected date has zero day_diffs; choose a later date.")

    fiber_path = str(
        req.get("fiber_path", "")
        or getattr(config, "mma_npy_path", "")
        or getattr(config, "fiber_csv_path", "")
    ).strip()
    if not fiber_path:
        raise ValueError("Fiber file path is empty.")
    resolver = getattr(config, "resolve_path", None)
    fiber_resolved = Path(resolver(fiber_path)) if callable(resolver) else Path(fiber_path)

    spacing_m = float(req.get("spacing_m", 0.1))
    origin_offset_m = float(req.get("origin_offset_m", -119.0))
    auto_origin = bool(req.get("auto_origin", True))
    reverse = bool(req.get("reverse", True))
    fiber_is_microstrain = bool(req.get("fiber_is_microstrain", True))
    b_min = float(req.get("b_min", 0.05))
    b_max = float(req.get("b_max", 2.0))
    b_step = float(req.get("b_step", 0.1))
    b_grid = _b_grid_from_step(b_min, b_max, b_step)

    fiber_x_base, fiber_y, x_from_file = load_fiber_yearly_profile(
        fiber_resolved,
        spacing_m=spacing_m,
        origin_offset_m=0.0,
        reverse=reverse,
        skip_header=bool(req.get("skip_header", True)),
        fiber_is_microstrain=fiber_is_microstrain,
    )

    prev_mode = str(getattr(config, "prediction_coordinate_system", "Fault-parallel/perpendicular"))
    config.prediction_coordinate_system = "Fault-parallel/perpendicular"
    ssi_on = bool(getattr(config, "ssi_enabled", False))
    _emit(
        progress,
        5,
        "Fiber calibration: grid-search Fault-parallel b via Step 3+4 kernel "
        f"(SSI={'on' if ssi_on else 'off'}; no b16/b50/b84). "
        "Each trial recomputes all epochs; "
        f"{'auto |strain| peak origin' if auto_origin else 'manual origin'}."
    )
    _emit(
        progress,
        6,
        "Limitation: 12EI/h^3 and soil springs stay at current SSI settings; this run only sweeps b.",
    )

    try:
        params_now = _current_fault_parallel_params(state, epoch_idx)
        b_current = float(params_now[1])
        model_x_current, model_y_current = _predict_yearly_profile(
            config,
            state,
            epoch_idx,
            state.get("fault_fit_override_by_epoch", {}) if isinstance(state.get("fault_fit_override_by_epoch", {}), dict) else {},
        )
        fiber_x_current, origin_current = _fiber_x_aligned(
            fiber_x_base,
            fiber_y,
            model_x_current,
            model_y_current,
            auto_origin=auto_origin,
            origin_offset_m=origin_offset_m,
            x_from_file=x_from_file,
        )
        try:
            rmse_current = rmse_on_overlap(model_x_current, model_y_current, fiber_x_current, fiber_y)
        except Exception:
            rmse_current = float("nan")

        b_tried = []
        rmse_tried = []
        best_b = None
        best_rmse = float("inf")
        best_xy = (model_x_current, model_y_current)
        best_fiber_x = fiber_x_current
        best_origin = origin_current
        n_grid = int(b_grid.size)
        for i, b_val in enumerate(b_grid):
            b_f = float(b_val)
            pct = 8 + int(82 * (i + 1) / max(n_grid, 1))
            _emit(progress, pct, f"Fiber calibration trial {i + 1}/{n_grid}: b={b_f:.6g}")
            overrides = _overrides_with_fault_parallel_b(state, epoch_idx, b_f)
            model_x, model_y = _predict_yearly_profile(config, state, epoch_idx, overrides)
            fiber_x_trial, origin_trial = _fiber_x_aligned(
                fiber_x_base,
                fiber_y,
                model_x,
                model_y,
                auto_origin=auto_origin,
                origin_offset_m=origin_offset_m,
                x_from_file=x_from_file,
            )
            rmse = rmse_on_overlap(model_x, model_y, fiber_x_trial, fiber_y)
            b_tried.append(b_f)
            rmse_tried.append(rmse)
            _emit(
                progress,
                pct,
                f"[Fiber calib] b={b_f:.6g} RMSE={rmse:.6g} με/year"
                + (f" origin={origin_trial:.4g} m" if auto_origin else ""),
            )
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_b = b_f
                best_xy = (model_x, model_y)
                best_fiber_x = fiber_x_trial
                best_origin = origin_trial

        if best_b is None:
            raise ValueError("Fiber calibration failed: no finite RMSE on the b grid.")
    finally:
        config.prediction_coordinate_system = prev_mode

    x_fiber_peak = peak_x(fiber_x_base, fiber_y)
    x_model_peak = peak_x(best_xy[0], best_xy[1])
    payload = {
        "fiber_x": np.asarray(best_fiber_x, dtype=float),
        "fiber_y": np.asarray(fiber_y, dtype=float),
        "model_x": np.asarray(best_xy[0], dtype=float),
        "model_y_optimal": np.asarray(best_xy[1], dtype=float),
        "model_x_current": np.asarray(model_x_current, dtype=float),
        "model_y_current": np.asarray(model_y_current, dtype=float),
        "b_optimal": float(best_b),
        "rmse_optimal": float(best_rmse),
        "b_current": float(b_current),
        "rmse_current": float(rmse_current) if np.isfinite(rmse_current) else float("nan"),
        "b_tried": np.asarray(b_tried, dtype=float),
        "rmse_tried": np.asarray(rmse_tried, dtype=float),
        "epoch_idx": int(epoch_idx),
        "a_fixed": float(params_now[0]),
        "c_fixed": float(params_now[2]),
        "d_fixed": float(params_now[3]),
        "ssi_enabled": bool(ssi_on),
        "fiber_path": str(fiber_resolved),
        "spacing_m": float(spacing_m),
        "origin_offset_m": float(best_origin),
        "auto_origin": bool(auto_origin),
        "x_fiber_peak_base": float(x_fiber_peak),
        "x_model_peak": float(x_model_peak),
        "reverse": bool(reverse),
    }
    out = dict(state)
    out["fiber_calibration"] = payload
    _emit(
        progress,
        100,
        f"Fiber calibration done: b_optimal={best_b:.6g}, RMSE={best_rmse:.6g} με/year "
        f"(current b={b_current:.6g}, origin={best_origin:.4g} m"
        f"{', auto |strain| peak' if auto_origin else ''}"
        f", x_fiber_peak={x_fiber_peak:.4g}, x_model_peak={x_model_peak:.4g}). "
        "Apply optimal b to write Fit override; re-run Step 3/4 to refresh main plots.",
    )
    return out
