from __future__ import annotations

import copy
import time
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import expit
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, nearest_points, substring, unary_union

from src.config import AppConfig
from src.core.fault_model import (
    _fit_piecewise_linear,
    _fit_sigmoid,
    compute_fault_distance_field,
    compute_signed_distance_to_polyline,
    fit_model,
    sigmoid,
)
from src.core.insar_3d import match_insar, solve_3d_displacement
from src.core.ssi_solver import SSIParams, solve_pipe_ssi_fault_components
from src.io.readers import load_inputs

_STEP1_CACHE_LOCK = Lock()
_STEP1_DATA_CACHE: Dict[tuple, Dict] = {}
_STEP1_MATCH_CACHE: Dict[tuple, Dict] = {}
_STEP1_NEZ_CACHE: Dict[tuple, Dict] = {}


def _safe_file_sig(path_str: str, resolver) -> tuple:
    p = resolver(path_str)
    try:
        stat = Path(p).stat()
        return (str(p), int(stat.st_size), int(stat.st_mtime_ns))
    except Exception:
        return (str(p), -1, -1)


def _build_step1_data_key(config: AppConfig) -> tuple:
    return (
        _safe_file_sig(config.asc_shp_path, config.resolve_path),
        _safe_file_sig(config.desc_shp_path, config.resolve_path),
        _safe_file_sig(config.combined_shp_path, config.resolve_path),
        _safe_file_sig(config.fault_line_shp_path, config.resolve_path),
        _safe_file_sig(config.ml_fault_polyline_npy, config.resolve_path),
        _safe_file_sig(config.pipe_axis_path, config.resolve_path),
        _safe_file_sig(config.mma_npy_path, config.resolve_path),
        _safe_file_sig(config.fiber_csv_path, config.resolve_path),
        str(getattr(config, "processing_crs", "EPSG:3857")).upper(),
        str(getattr(config, "pipeline_input_crs", "EPSG:3857")).upper(),
        bool(getattr(config, "auto_use_ml_fault_polyline", True)),
    )


def _build_step1_match_key(config: AppConfig, data_key: tuple) -> tuple:
    return (
        data_key,
        int(getattr(config, "date_match_max_days", 4)),
        float(getattr(config, "spatial_match_max_m", 5.0)),
        str(getattr(config, "date_column_prefix", "D")),
    )


def _build_step1_nez_key(config: AppConfig, match_key: tuple, effective_aspect: float) -> tuple:
    return (
        match_key,
        float(getattr(config, "asc_heading_deg", np.nan)),
        float(getattr(config, "asc_inc_deg", np.nan)),
        float(getattr(config, "desc_heading_deg", np.nan)),
        float(getattr(config, "desc_inc_deg", np.nan)),
        str(getattr(config, "constraint_mode", "aspect_direction")),
        float(effective_aspect),
    )


def _bearing_from_line_north_clockwise(line: LineString) -> float:
    coords = np.asarray(line.coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return np.nan
    p0 = coords[0, :2]
    p1 = coords[-1, :2]
    vec = p1 - p0
    if np.linalg.norm(vec) <= 1e-12:
        return np.nan
    dx = float(vec[0])
    dy = float(vec[1])
    # Bearing: north(+Y)=0, clockwise positive.
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def compute_unit_tangent_and_normal(points: np.ndarray, polyline: np.ndarray):
    pts = np.asarray(points, dtype=float)
    pl = np.asarray(polyline, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] < 2:
        tangent = np.tile(np.array([[1.0, 0.0]], dtype=float), (pts.shape[0], 1))
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        return tangent, normal

    seg_start = pl[:-1, :2]
    seg_end = pl[1:, :2]
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid = seg_len > 1e-12
    if not np.any(valid):
        tangent = np.tile(np.array([[1.0, 0.0]], dtype=float), (pts.shape[0], 1))
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        return tangent, normal
    seg_start = seg_start[valid]
    seg_vec = seg_vec[valid]
    seg_len = seg_len[valid]
    seg_dir = seg_vec / seg_len[:, None]

    out_tangent = np.zeros((pts.shape[0], 2), dtype=float)
    chunk = 2048
    for beg in range(0, pts.shape[0], chunk):
        end = min(pts.shape[0], beg + chunk)
        pblk = pts[beg:end, :2]
        rel = pblk[:, None, :] - seg_start[None, :, :]
        proj_len = np.einsum("bij,ij->bi", rel, seg_dir)
        proj_len = np.clip(proj_len, 0.0, seg_len[None, :])
        proj = seg_start[None, :, :] + proj_len[:, :, None] * seg_dir[None, :, :]
        delta = pblk[:, None, :] - proj
        d2 = np.einsum("bij,bij->bi", delta, delta)
        best = np.argmin(d2, axis=1)
        out_tangent[beg:end] = seg_dir[best]

    out_normal = np.column_stack((-out_tangent[:, 1], out_tangent[:, 0]))
    return out_tangent, out_normal


def compute_xy_displacement(normals, tangents, disp_normals, disp_tangents):
    normals = np.asarray(normals, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    disp_normals = np.asarray(disp_normals).reshape(-1, 1)
    disp_tangents = np.asarray(disp_tangents).reshape(-1, 1)
    # Normalize basis vectors defensively to avoid projection-scale drift.
    tn = np.linalg.norm(tangents, axis=1, keepdims=True)
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    tn = np.where(tn <= 1e-12, 1.0, tn)
    nn = np.where(nn <= 1e-12, 1.0, nn)
    tangents = tangents / tn
    normals = normals / nn
    displacements = disp_normals * normals + disp_tangents * tangents
    return displacements[:, 0], displacements[:, 1]


def curvature_2d(p_prev, p, p_next):
    x1, y1 = p_prev
    x2, y2 = p
    x3, y3 = p_next
    num = 2.0 * ((x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2))
    a = np.linalg.norm(p_prev - p)
    b = np.linalg.norm(p_next - p)
    c = np.linalg.norm(p_next - p_prev)
    denom = a * b * c
    return num / denom if denom != 0 else 0.0


def compute_relative_bending_strain(new_xy, new_xy_deformed, diameter_m: float = 0.9144):
    n = len(new_xy)
    strain = np.zeros(n)
    diameter = float(diameter_m) if np.isfinite(diameter_m) and float(diameter_m) > 0.0 else 0.9144
    for i in range(1, n - 1):
        k0 = curvature_2d(new_xy[i - 1], new_xy[i], new_xy[i + 1])
        k1 = curvature_2d(new_xy_deformed[i - 1], new_xy_deformed[i], new_xy_deformed[i + 1])
        strain[i] = (k1 - k0) * (diameter / 2.0)
    if n > 1:
        strain[0] = strain[1]
        strain[-1] = strain[-2]
    return strain


def _pick_intersection_point(pipe_line: LineString, fault_line: LineString) -> np.ndarray:
    inter = pipe_line.intersection(fault_line)
    if inter.is_empty:
        p_pipe, _ = nearest_points(pipe_line, fault_line)
        return np.array([p_pipe.x, p_pipe.y], dtype=float)
    if inter.geom_type == "Point":
        return np.array([inter.x, inter.y], dtype=float)
    if hasattr(inter, "geoms"):
        for g in inter.geoms:
            if g.geom_type == "Point":
                return np.array([g.x, g.y], dtype=float)
    p_pipe, _ = nearest_points(pipe_line, fault_line)
    return np.array([p_pipe.x, p_pipe.y], dtype=float)


def _compute_fit_band_mask(
    config: AppConfig,
    points: np.ndarray,
    fault_line: LineString,
    pipe_axis: np.ndarray,
    intersection_override: Optional[np.ndarray] = None,
):
    """
    Follow notebook cells 32-33:
    - build perpendicular transect at pipe-fault intersection
    - create two offset lines
    - use polygon between offset lines as fitting band
    """
    pipe_line = LineString(pipe_axis)
    intersection = (
        np.asarray(intersection_override, dtype=float).reshape(-1)[:2]
        if intersection_override is not None
        else _pick_intersection_point(pipe_line, fault_line)
    )
    fault_coords = np.asarray(fault_line.coords, dtype=float)
    if fault_coords.ndim != 2 or fault_coords.shape[0] < 2:
        tangent = np.array([1.0, 0.0], dtype=float)
    else:
        ref_pt = Point(float(intersection[0]), float(intersection[1]))
        s_proj = float(fault_line.project(ref_pt))
        seg_vec = np.diff(fault_coords[:, :2], axis=0)
        seg_len = np.linalg.norm(seg_vec, axis=1)
        cum = np.insert(np.cumsum(seg_len), 0, 0.0)
        idx_fault = int(np.clip(np.searchsorted(cum, s_proj, side="right") - 1, 0, len(seg_len) - 1))
        if seg_len[idx_fault] <= 1e-12:
            nz = np.where(seg_len > 1e-12)[0]
            if nz.size > 0:
                idx_fault = int(nz[np.argmin(np.abs(cum[nz] - s_proj))])
        seg = seg_vec[idx_fault]
        seg_norm = np.linalg.norm(seg)
        tangent = seg / seg_norm if seg_norm > 0 else np.array([1.0, 0.0], dtype=float)
    normal = np.array([-tangent[1], tangent[0]])

    length = float(config.transect_half_length_m)
    p_start = intersection - normal * length
    p_end = intersection + normal * length

    # Offset transects along fault tangent direction.
    # This is the range selector for InSAR points used in fitting.
    off = float(config.fit_band_offset_m)
    line1_start = p_start - tangent * off
    line1_end = p_end - tangent * off
    line2_start = p_start + tangent * off
    line2_end = p_end + tangent * off

    # Build cap lines at transect half-length endpoints (perpendicular to transect).
    cap_start_a = line1_start
    cap_start_b = line2_start
    cap_end_a = line1_end
    cap_end_b = line2_end

    def _line_intersection(pa, pb, qa, qb):
        pa = np.asarray(pa, dtype=float)
        pb = np.asarray(pb, dtype=float)
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)
        r = pb - pa
        s = qb - qa
        den = float(r[0] * s[1] - r[1] * s[0])
        if abs(den) <= 1e-12:
            return None
        qp = qa - pa
        t = float((qp[0] * s[1] - qp[1] * s[0]) / den)
        return pa + t * r

    v1 = _line_intersection(line1_start, line1_end, cap_start_a, cap_start_b)  # offset_neg ∩ cap_start
    v2 = _line_intersection(line1_start, line1_end, cap_end_a, cap_end_b)      # offset_neg ∩ cap_end
    v3 = _line_intersection(line2_start, line2_end, cap_end_a, cap_end_b)      # offset_pos ∩ cap_end
    v4 = _line_intersection(line2_start, line2_end, cap_start_a, cap_start_b)  # offset_pos ∩ cap_start
    quad_fallback = any(v is None for v in (v1, v2, v3, v4))
    if quad_fallback:
        # Fallback to endpoint stitching when near-parallel numerical instability occurs.
        pg_vertices = np.asarray([line1_start, line1_end, line2_end, line2_start], dtype=float)
    else:
        pg_vertices = np.asarray([v1, v2, v3, v4], dtype=float)

    pg = Polygon(pg_vertices)
    if not pg.is_valid:
        pg = pg.buffer(0)
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 2:
        # Use shapely polygon-point test directly so all points inside the fit quadrilateral
        # (including points on boundary) are selected as fit subset.
        mask = np.array([pg.covers(Point(float(p[0]), float(p[1]))) for p in pts[:, :2]], dtype=bool)
    else:
        mask = np.zeros(0, dtype=bool)
    return {
        "mask": mask,
        "polygon": pg,
        "polygon_vertices": pg_vertices,
        "quad_fallback": bool(quad_fallback),
        "intersection": intersection,
        "transect": LineString([p_start, p_end]),
        "offset_lines": (LineString([line1_start, line1_end]), LineString([line2_start, line2_end])),
        "cap_lines": (LineString([cap_start_a, cap_start_b]), LineString([cap_end_a, cap_end_b])),
    }


def _to_linestring(geom) -> LineString:
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        return max(list(geom.geoms), key=lambda g: g.length)
    raise ValueError(f"Unsupported geometry type for line conversion: {getattr(geom, 'geom_type', type(geom))}")


def _resample_polyline_by_spacing(points: np.ndarray, spacing_m: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return pts
    seg = np.diff(pts[:, :2], axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.insert(np.cumsum(seg_len), 0, 0.0)
    total = float(cum[-1])
    if total <= 0:
        return pts
    spacing = max(float(spacing_m), 0.1)
    n_steps = max(2, int(np.ceil(total / spacing)) + 1)
    target = np.linspace(0.0, total, n_steps)
    sampled = np.zeros((n_steps, pts.shape[1]), dtype=float)
    for i, td in enumerate(target):
        si = int(np.clip(np.searchsorted(cum, td) - 1, 0, len(seg_len) - 1))
        base = float(cum[si])
        sl = float(seg_len[si])
        t = (td - base) / sl if sl > 0 else 0.0
        sampled[i] = (1.0 - t) * pts[si] + t * pts[si + 1]
    return sampled


def _as_lines_list(geom) -> list:
    if geom is None:
        return []
    if isinstance(geom, LineString):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiLineString):
        return [ln for ln in geom.geoms if ln is not None and not ln.is_empty]
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_as_lines_list(g))
        return out
    return []


def _merge_lines_to_single(lines: list, label: str, require_connected: bool) -> LineString:
    if not lines:
        raise ValueError(f"{label} has no valid line geometry.")
    if len(lines) == 1:
        return lines[0]
    merged = linemerge(unary_union(lines))
    if isinstance(merged, LineString):
        return merged
    if isinstance(merged, MultiLineString):
        parts = [ln for ln in merged.geoms if ln is not None and not ln.is_empty]
        if not parts:
            raise ValueError(f"{label} could not be merged into a valid line.")
        if require_connected and len(parts) > 1:
            raise ValueError(f"{label} selected segments are not connected.")
        return max(parts, key=lambda g: g.length)
    raise ValueError(f"{label} merged geometry is not a LineString.")


def _build_pipe_segments_from_selected(data: Dict, selected_pipeline_idxs) -> list[np.ndarray]:
    lines = []
    for geom in data.get("pipe_lines_all", []):
        lines.extend(_as_lines_list(geom))
    if selected_pipeline_idxs is not None and len(selected_pipeline_idxs) > 0:
        idxs = sorted(set(int(i) for i in selected_pipeline_idxs))
        bad = [i for i in idxs if i < 0 or i >= len(lines)]
        if bad:
            raise ValueError(f"Invalid selected pipeline segment index: {bad}")
        chosen = [lines[i] for i in idxs]
    else:
        axis = np.asarray(data.get("pipe_axis", []), dtype=float)
        if axis.ndim == 2 and axis.shape[0] > 1 and axis.shape[1] >= 2:
            return [axis[:, :2]]
        if not lines:
            raise ValueError("No valid pipeline geometry available.")
        chosen = lines
    segments = []
    for ln in chosen:
        coords = np.asarray(ln.coords, dtype=float)[:, :2]
        if coords.shape[0] >= 2:
            segments.append(coords)
    if not segments:
        raise ValueError("Selected pipeline geometry has insufficient nodes.")
    return segments


def _flatten_pipe_segments(pipe_segments: list[np.ndarray]) -> np.ndarray:
    if not pipe_segments:
        return np.empty((0, 2), dtype=float)
    merged = []
    for i, seg in enumerate(pipe_segments):
        arr = np.asarray(seg, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        if i > 0:
            merged.append(np.array([[np.nan, np.nan]], dtype=float))
        merged.append(arr[:, :2])
    if not merged:
        return np.empty((0, 2), dtype=float)
    return np.vstack(merged).astype(float)


def _select_pipe_segment_near_center(pipe_segments: list[np.ndarray], center_xy: np.ndarray) -> np.ndarray:
    if not pipe_segments:
        return np.empty((0, 2), dtype=float)
    cpt = Point(float(center_xy[0]), float(center_xy[1]))
    best_idx = 0
    best_dist = float("inf")
    for i, seg in enumerate(pipe_segments):
        arr = np.asarray(seg, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        dist = float(LineString(arr[:, :2]).distance(cpt))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return np.asarray(pipe_segments[best_idx], dtype=float)[:, :2]


def _fit_fault_line_by_degree(base_line: LineString, degree: int, sample_count: int) -> tuple[LineString, Dict]:
    coords = np.asarray(base_line.coords, dtype=float)
    if coords.shape[0] < degree + 1:
        raise ValueError("Fault fit requires more points. Please select longer fault segments.")
    seg = np.diff(coords[:, :2], axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Fault fit failed: selected fault geometry length is zero.")
    t = s / total
    deg = int(np.clip(degree, 1, min(5, coords.shape[0] - 1)))
    coef_x = np.polyfit(t, coords[:, 0], deg).astype(float)
    coef_y = np.polyfit(t, coords[:, 1], deg).astype(float)
    n_samp = max(int(sample_count), 100)
    tt = np.linspace(0.0, 1.0, n_samp)
    fit_xy = np.column_stack([np.polyval(coef_x, tt), np.polyval(coef_y, tt)])
    fit_line = LineString(fit_xy)
    return fit_line, {
        "fit_degree": deg,
        "fit_coefficients": {
            "x(t)": coef_x.tolist(),
            "y(t)": coef_y.tolist(),
        },
        "fit_sample_count": int(n_samp),
        "fit_base_length_m": float(total),
    }


def _build_extended_fitted_fault_line(
    fit_meta: Dict,
    base_length_m: float,
    default_sample_count: int,
    min_extend_m: float = 1000.0,
    extend_ratio_each_side: float = 0.5,
) -> tuple[Optional[LineString], Dict]:
    coeff = fit_meta.get("fit_coefficients", {}) if isinstance(fit_meta, dict) else {}
    cx = np.asarray(coeff.get("x(t)", []), dtype=float).reshape(-1)
    cy = np.asarray(coeff.get("y(t)", []), dtype=float).reshape(-1)
    if cx.size < 2 or cy.size < 2:
        return None, {"extended_available": False}
    base_len = float(base_length_m) if np.isfinite(base_length_m) and base_length_m > 1e-9 else 1.0
    extra_frac = max(float(extend_ratio_each_side), float(min_extend_m) / base_len)
    t0 = -extra_frac
    t1 = 1.0 + extra_frac
    n_base = max(int(default_sample_count), int(fit_meta.get("fit_sample_count", default_sample_count)), 120)
    n_ext = max(int(np.ceil(n_base * (1.0 + 2.0 * extra_frac))), n_base + 40)
    tt = np.linspace(t0, t1, n_ext)
    xy = np.column_stack([np.polyval(cx, tt), np.polyval(cy, tt)])
    keep = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    xy = xy[keep]
    if xy.shape[0] < 2:
        return None, {"extended_available": False}
    ext_line = LineString(xy)
    return ext_line, {
        "extended_available": True,
        "extend_t_start": float(t0),
        "extend_t_end": float(t1),
        "extend_ratio_each_side": float(extra_frac),
        "extend_min_m": float(min_extend_m),
        "extended_sample_count": int(xy.shape[0]),
    }


def _build_fault_trace_from_selected(
    config: AppConfig,
    fault_data,
    selected_fault_idxs,
) -> tuple[LineString, LineString, Optional[LineString], Optional[LineString], Dict]:
    all_lines = []
    for geom in fault_data.geometry.values:
        all_lines.extend(_as_lines_list(geom))
    if not all_lines:
        raise ValueError("fault_data contains no valid line geometry.")
    if selected_fault_idxs is not None and len(selected_fault_idxs) > 0:
        idxs = sorted(set(int(i) for i in selected_fault_idxs))
        bad = [i for i in idxs if i < 0 or i >= len(all_lines)]
        if bad:
            raise ValueError(f"Invalid selected fault segment index: {bad}")
        chosen = [all_lines[i] for i in idxs]
    else:
        idxs = list(range(len(all_lines)))
        chosen = all_lines
    trace_type = str(getattr(config, "fault_trace_type", "selected_geometry"))
    summary = {
        "trace_type": trace_type,
        "selected_fault_count": int(len(idxs)),
        "selected_fault_idxs": idxs,
    }
    selected_line = _merge_lines_to_single(chosen, "Fault", require_connected=False)
    if trace_type == "selected_geometry":
        fault_line = _merge_lines_to_single(chosen, "Fault", require_connected=True)
        return fault_line, selected_line, None, None, summary
    base_line = selected_line
    degree = 1 if trace_type == "linear_fit" else 2
    fit_line, fit_meta = _fit_fault_line_by_degree(
        base_line=base_line,
        degree=degree,
        sample_count=int(getattr(config, "fault_fit_sample_count", 900)),
    )
    fit_ext, ext_meta = _build_extended_fitted_fault_line(
        fit_meta=fit_meta,
        base_length_m=float(base_line.length),
        default_sample_count=int(getattr(config, "fault_fit_sample_count", 900)),
        min_extend_m=1000.0,
        extend_ratio_each_side=0.5,
    )
    summary.update(fit_meta)
    summary.update(ext_meta)
    return fit_line, selected_line, fit_line, fit_ext, summary


def _resolve_effective_constraint_aspect(
    config: AppConfig,
    fault_data,
    selected_fault_idxs,
    fallback_line: Optional[LineString] = None,
) -> tuple[float, str]:
    source = str(getattr(config, "constraint_aspect_source", "manual")).strip().lower()
    manual = float(getattr(config, "constraint_aspect_deg", 0.0))
    if source != "from_selected_fault_linear_fit":
        return manual, "manual"

    all_lines = []
    for geom in fault_data.geometry.values:
        all_lines.extend(_as_lines_list(geom))
    if not all_lines:
        return manual, "manual_fallback_no_fault_lines"
    idxs = (
        sorted(set(int(i) for i in selected_fault_idxs))
        if (selected_fault_idxs is not None and len(selected_fault_idxs) > 0)
        else list(range(len(all_lines)))
    )
    idxs = [i for i in idxs if 0 <= i < len(all_lines)]
    if not idxs:
        return manual, "manual_fallback_no_selected_fault"
    chosen = [all_lines[i] for i in idxs]
    try:
        base_line = _merge_lines_to_single(chosen, "Fault", require_connected=False)
        fit_line, _ = _fit_fault_line_by_degree(
            base_line=base_line,
            degree=1,
            sample_count=max(int(getattr(config, "fault_fit_sample_count", 900)), 100),
        )
        aspect = _bearing_from_line_north_clockwise(fit_line)
        if np.isfinite(aspect):
            return float(aspect), "from_selected_fault_linear_fit"
    except Exception:
        pass
    if isinstance(fallback_line, LineString):
        aspect_fb = _bearing_from_line_north_clockwise(fallback_line)
        if np.isfinite(aspect_fb):
            return float(aspect_fb), "manual_fallback_effective_faultline"
    return manual, "manual_fallback_fit_failed"


def _extract_intersection_points(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return [g for g in geom.geoms if g is not None and not g.is_empty]
    if geom.geom_type == "LineString":
        return [geom.interpolate(0.5, normalized=True)]
    if geom.geom_type == "MultiLineString":
        out = []
        for ln in geom.geoms:
            if ln is None or ln.is_empty:
                continue
            out.append(ln.interpolate(0.5, normalized=True))
        return out
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(_extract_intersection_points(g))
        return out
    return []


def _compute_intersection_center(
    pipe_segments: list[np.ndarray],
    fault_line: LineString,
    trace_type: str = "selected_geometry",
) -> Dict:
    points = []
    for seg in pipe_segments:
        arr = np.asarray(seg, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        pipe_line = LineString(arr[:, :2])
        inter = pipe_line.intersection(fault_line)
        points.extend(_extract_intersection_points(inter))
    if not points:
        msg = "No intersection between selected pipeline and fault trace."
        if str(trace_type) in {"linear_fit", "quadratic_fit"}:
            msg = "No intersection between selected pipeline and fitted fault trace."
        raise ValueError(
            f"{msg} "
            "Please reselect segments or adjust fault trace type."
        )
    pts_xy = np.array([[float(pt.x), float(pt.y)] for pt in points], dtype=float)
    if pts_xy.shape[0] == 1:
        center_xy = pts_xy[0]
        method = "single-intersection"
    else:
        center_xy = np.mean(pts_xy, axis=0)
        method = "mean-of-intersections"
    return {
        "center_xy": np.asarray(center_xy, dtype=float),
        "intersection_count": int(pts_xy.shape[0]),
        "intersection_points_xy": pts_xy.tolist(),
        "intersection_center_method": method,
    }


def _resolve_intersection_center_with_trace_fallback(
    pipe_segments: list[np.ndarray],
    trace_type: str,
    fitted_raw: Optional[LineString],
    fitted_extended: Optional[LineString],
    default_trace: LineString,
) -> tuple[Dict, LineString, str]:
    tt = str(trace_type)
    if tt not in {"linear_fit", "quadratic_fit"}:
        info = _compute_intersection_center(pipe_segments, default_trace, trace_type=tt)
        return info, default_trace, "selected_or_default"
    if isinstance(fitted_raw, LineString):
        try:
            info = _compute_intersection_center(pipe_segments, fitted_raw, trace_type=tt)
            return info, fitted_raw, "raw_fit"
        except Exception:
            pass
    if isinstance(fitted_extended, LineString):
        try:
            info = _compute_intersection_center(pipe_segments, fitted_extended, trace_type=tt)
            return info, fitted_extended, "extended_fit"
        except Exception:
            pass
    raise ValueError(
        "No intersection between selected pipeline and fitted (extended) fault trace. "
        "Please reselect segments or adjust fault trace type."
    )


def _clip_single_pipe_segment_around_center(
    pipe_axis: np.ndarray,
    center_xy: np.ndarray,
    half_length_m: float,
) -> tuple[np.ndarray, Dict]:
    line = LineString(np.asarray(pipe_axis, dtype=float)[:, :2])
    total = float(line.length)
    if total <= 0:
        raise ValueError("Pipeline geometry length is zero.")
    center = np.asarray(center_xy, dtype=float).reshape(-1)[:2]
    s0 = float(line.project(Point(float(center[0]), float(center[1]))))
    half = max(float(half_length_m), 1.0)
    s_start = max(0.0, s0 - half)
    s_end = min(total, s0 + half)
    if s_end - s_start <= 1e-6:
        raise ValueError("Pipeline clipping range is too short; increase clip half-length.")
    clipped = substring(line, s_start, s_end, normalized=False)
    if isinstance(clipped, MultiLineString):
        parts = [ln for ln in clipped.geoms if ln is not None and not ln.is_empty]
        if not parts:
            raise ValueError("Pipeline clipping produced empty geometry.")
        clipped = max(parts, key=lambda g: g.length)
    if not isinstance(clipped, LineString):
        raise ValueError("Pipeline clipping did not produce a valid line.")
    coords = np.asarray(clipped.coords, dtype=float)
    if coords.shape[0] < 2:
        raise ValueError("Pipeline clipping produced insufficient nodes.")
    return coords[:, :2], {
        "segment_length_before_clip_m": total,
        "segment_length_after_clip_m": float(clipped.length),
        "segment_nodes_before_clip": int(np.asarray(pipe_axis).shape[0]),
        "segment_nodes_after_clip": int(coords.shape[0]),
        "pipe_nodes_after_clip": int(coords.shape[0]),
        "clip_center_s_m": s0,
        "clip_start_s_m": s_start,
        "clip_end_s_m": s_end,
    }


def _clip_pipe_segments_around_center(
    pipe_segments: list[np.ndarray],
    center_xy: np.ndarray,
    half_length_m: float,
) -> tuple[list[np.ndarray], Dict]:
    clipped_segments = []
    before_lengths = []
    after_lengths = []
    for seg in pipe_segments:
        arr = np.asarray(seg, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        clip_seg, seg_summary = _clip_single_pipe_segment_around_center(arr[:, :2], center_xy, half_length_m)
        clipped_segments.append(clip_seg)
        before_lengths.append(float(seg_summary.get("segment_length_before_clip_m", np.nan)))
        after_lengths.append(float(seg_summary.get("segment_length_after_clip_m", np.nan)))
    if not clipped_segments:
        raise ValueError("Pipeline clipping produced empty geometry.")
    total_before = float(np.nansum(before_lengths)) if before_lengths else np.nan
    total_after = float(np.nansum(after_lengths)) if after_lengths else np.nan
    summary = {
        "pipe_segments_before_clip": int(len(pipe_segments)),
        "pipe_segments_after_clip": int(len(clipped_segments)),
        "pipe_length_before_clip_m": total_before,
        "pipe_length_after_clip_m": total_after,
        "pipe_nodes_before_clip": int(sum(np.asarray(seg).shape[0] for seg in pipe_segments)),
        "pipe_nodes_after_clip": int(sum(np.asarray(seg).shape[0] for seg in clipped_segments)),
    }
    return clipped_segments, summary


def _compute_step1_auto_view(pipe_axis: np.ndarray, center_xy: np.ndarray) -> Dict:
    pts = np.asarray(pipe_axis, dtype=float)
    cx = float(center_xy[0])
    cy = float(center_xy[1])
    if pts.ndim != 2 or pts.shape[0] < 2:
        half = 300.0
    else:
        x_span = float(np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0]))
        y_span = float(np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1]))
        span = max(x_span, y_span, 50.0)
        half = max(1.2 * span, 120.0)
    return {
        "xlim": [cx - half, cx + half],
        "ylim": [cy - half, cy + half],
        "center_xy": [cx, cy],
    }


def _build_notebook_style_fault_refs(config: AppConfig, fault_data, pipe_axis: np.ndarray, points: np.ndarray):
    """
    Reproduce notebook cells 27-31:
    - fault_line_truth: intersecting fault geometry (notebook used index 6)
    - connector: extra segment (notebook used index 213)
    - fl2 = connector + fault_line_truth (for tangent/intersection/band)
    - distances_truth computed by nearest-segment signed distance (right+, left-).
    """
    pipe_line = LineString(pipe_axis)
    geoms = list(fault_data.geometry.values)
    if not geoms:
        raise ValueError("fault_data is empty.")
    lines = [_to_linestring(g) for g in geoms]

    intersect_ids = [i for i, g in enumerate(lines) if g.intersects(pipe_line)]
    truth_idx = intersect_ids[0] if intersect_ids else int(np.argmax([g.length for g in lines]))
    fault_line_truth = lines[truth_idx]

    connector_idx = 213 if len(lines) > 213 else None
    if connector_idx is None or connector_idx == truth_idx:
        # fallback: pick the closest non-truth line to mimic notebook's connected extension behavior.
        cand = [(i, g.distance(fault_line_truth)) for i, g in enumerate(lines) if i != truth_idx]
        connector_idx = min(cand, key=lambda t: t[1])[0] if cand else truth_idx
    connector_line = lines[connector_idx]

    connector_coords = np.asarray(connector_line.coords)
    truth_coords = np.asarray(fault_line_truth.coords)
    fl2 = np.concatenate([connector_coords, truth_coords], axis=0)
    fault_line_truth2 = LineString(fl2)

    # Use segment-projection signed distance. This avoids polygon side masks.
    distances_truth = compute_signed_distance_to_polyline(
        points=np.asarray(points, dtype=float),
        polyline_coords=truth_coords,
        flip_sign=bool(getattr(config, "flip_signed_distance", False)),
    )

    return {
        "fault_line_truth": fault_line_truth,
        "fault_line_truth2": fault_line_truth2,
        "fl2": fl2,
        "distances_truth": distances_truth,
        "left_area": None,
        "truth_idx": truth_idx,
        "connector_idx": connector_idx,
    }


def _evaluate_shape_function(distance: np.ndarray, fit_result, model_type: str) -> np.ndarray:
    if model_type == "sigmoid":
        return sigmoid(distance, *_normalize_sigmoid_params(np.asarray(fit_result.params, dtype=float)))
    params = np.asarray(fit_result.params, dtype=float)
    return np.where(distance < 0, params[0] * distance + params[1], params[2] * distance + params[3])


def _normalize_sigmoid_params(params_like) -> np.ndarray:
    arr = np.asarray(params_like, dtype=float).reshape(-1)
    if arr.size >= 4:
        return np.array(arr[:4], dtype=float, copy=True)
    if arr.size == 3:
        return np.array([arr[0], arr[1], 0.0, arr[2]], dtype=float)
    if arr.size > 0:
        out = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        out[: min(arr.size, 4)] = arr[: min(arr.size, 4)]
        return out
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)


def _resolve_fit_params_with_override(fit_result, override_values):
    base = _normalize_sigmoid_params(np.asarray(fit_result.params, dtype=float))
    if override_values is None:
        return np.array(base, dtype=float, copy=True)
    arr = np.asarray(override_values, dtype=float).reshape(-1)
    if arr.size >= 4:
        return arr[:4]
    if arr.size == 3:
        return np.array([arr[0], arr[1], 0.0, arr[2]], dtype=float)
    return base


def _compute_shear_band_width_from_b(b_value: float) -> float:
    if not np.isfinite(b_value) or abs(float(b_value)) < 1e-12:
        return float("inf")
    x_1 = np.log(0.01 / 0.99) / float(b_value)
    x_99 = np.log(0.99 / 0.01) / float(b_value)
    return abs(float(x_99 - x_1))


def _sigmoid_transition_zone_from_b(b_value: float, center_x: float = 0.0) -> Dict[str, float]:
    b = float(b_value)
    if not np.isfinite(b) or abs(b) < 1e-12:
        return {"x01": np.nan, "x99": np.nan, "width": np.nan, "center": float(center_x)}
    x01 = float(center_x + np.log(0.01 / 0.99) / b)
    x99 = float(center_x + np.log(0.99 / 0.01) / b)
    return {
        "x01": min(x01, x99),
        "x99": max(x01, x99),
        "width": abs(float(x99 - x01)),
        "center": float(center_x),
    }


def _build_ssi_params_from_config(config: AppConfig) -> SSIParams:
    return SSIParams(
        D_out_m=float(getattr(config, "ssi_D_out_m", 0.9144)),
        t_wall_m=float(getattr(config, "ssi_t_wall_m", 0.08313)),
        E_pipe_kN_m2=float(getattr(config, "ssi_E_pipe_kN_m2", 1000000.0)),
        bc_type=str(getattr(config, "ssi_bc_type", "pinned-pinned")),
        E_soil_outer_kN_m2=float(getattr(config, "ssi_E_soil_outer_kN_m2", 10000.0)),
        E_soil_center_kN_m2=float(getattr(config, "ssi_E_soil_center_kN_m2", 2000.0)),
        k2_ratio_outer=float(getattr(config, "ssi_k2_ratio_outer", 0.325)),
        k2_ratio_center=float(getattr(config, "ssi_k2_ratio_center", 0.325)),
        mu_friction=float(getattr(config, "ssi_mu_friction", 0.678)),
        delta_y_m=float(getattr(config, "ssi_delta_y_m", 0.001)),
        delta_max_m=float(getattr(config, "ssi_delta_max_m", 0.023)),
        k_axial_kN_m=float(getattr(config, "ssi_k_axial_kN_m", 10000.0)),
        depth_burial_m=float(getattr(config, "ssi_depth_burial_m", 1.5)),
        gamma_soil_kN_m3=float(getattr(config, "ssi_gamma_soil_kN_m3", 18.0)),
        k_gap_penalty_kN_m=float(getattr(config, "ssi_k_gap_penalty_kN_m", 5.0e5)),
        n_increments=int(getattr(config, "ssi_n_increments", 20)),
        max_iter=int(getattr(config, "ssi_max_iter", 50)),
        tol=float(getattr(config, "ssi_tol", 1e-6)),
    )


def _safe_sigma_b_from_fit(fit_result) -> float:
    pcov = getattr(fit_result, "pcov", None)
    if pcov is None:
        return np.nan
    cov = np.asarray(pcov, dtype=float)
    if cov.shape[0] <= 1 or cov.shape[1] <= 1:
        return np.nan
    var_b = float(cov[1, 1])
    if not np.isfinite(var_b) or var_b < 0:
        return np.nan
    return float(np.sqrt(var_b))


def _build_epistemic_b_entry(fit_result, b16_floor: float = 0.0) -> Dict:
    params = np.asarray(fit_result.params, dtype=float).reshape(-1)
    b50 = float(params[1]) if params.size >= 2 else np.nan
    sigma_b = _safe_sigma_b_from_fit(fit_result)
    b16 = b50 - sigma_b if np.isfinite(sigma_b) else np.nan
    b84 = b50 + sigma_b if np.isfinite(sigma_b) else np.nan
    floor = float(b16_floor)
    b16_disabled = bool(np.isfinite(b16) and b16 <= floor)
    if b16_disabled:
        b16 = np.nan
    return {
        "b50": b50,
        "sigma_b": sigma_b,
        "b16": b16,
        "b84": b84,
        "b16_floor": floor,
        "b16_disabled": b16_disabled,
        "shear_band_width_m": _compute_shear_band_width_from_b(b50),
    }


def _build_fault_parallel_overrides_for_case(state: Dict, pair_count: int, case_key: str) -> Dict[int, Dict[str, np.ndarray]]:
    out = {}
    epi = state.get("epistemic_b_by_epoch", {})
    user_overrides = state.get("fault_fit_override_by_epoch", {})
    for i in range(pair_count):
        fit_t = state.get("tangential_fit_by_epoch", {}).get(i)
        fit_n = state.get("normal_fit_by_epoch", {}).get(i)
        if fit_t is None or fit_n is None:
            continue
        epoch_override = user_overrides.get(i, {}) if isinstance(user_overrides, dict) else {}
        p_t = _resolve_fit_params_with_override(fit_t, epoch_override.get("Fault-parallel"))
        p_n = _resolve_fit_params_with_override(fit_n, epoch_override.get("Fault-perpendicular"))
        b_info = epi.get(i, {})
        if case_key == "b16" and bool(b_info.get("b16_disabled", False)):
            out[i] = {
                "Fault-parallel": p_t[:4],
                "Fault-perpendicular": p_n[:4],
            }
            continue
        if case_key in {"b16", "b84"}:
            b_case = float(b_info.get(case_key, np.nan))
            if np.isfinite(b_case) and p_t.size >= 2:
                p_t[1] = b_case
        out[i] = {
            "Fault-parallel": p_t[:4],
            "Fault-perpendicular": p_n[:4],
        }
    return out


def _build_ne_models_for_case(state: Dict, pair_count: int, case_key: str) -> Dict[int, Dict]:
    out = {}
    epi_ne = state.get("epistemic_b_ne_by_epoch", {})
    for i in range(pair_count):
        model_i = state.get("models_by_epoch", {}).get(i)
        if model_i is None:
            continue
        model_case = copy.deepcopy(model_i)
        ne_entry = epi_ne.get(i, {})
        for direction, comp in (("North", "x_component"), ("East", "y_component")):
            entry = ne_entry.get(direction, {})
            p = np.asarray(model_case[comp].params, dtype=float).copy()
            if case_key == "b16" and bool(entry.get("b16_disabled", False)):
                model_case[comp].params = p
                continue
            b_case = float(entry.get(case_key, np.nan))
            if np.isfinite(b_case) and p.size >= 2:
                p[1] = b_case
            model_case[comp].params = p
        out[i] = model_case
    return out


def _run_epistemic_cases(
    config: AppConfig,
    state: Dict,
    pair_count: int,
    include_strain: bool,
    prediction_mode: Optional[str] = None,
) -> Dict:
    data = state["data"]
    cases = {}
    mode = prediction_mode or getattr(config, "prediction_coordinate_system", "Fault-parallel/perpendicular")
    include_b16 = False
    if mode == "N&E":
        epi_ne = state.get("epistemic_b_ne_by_epoch", {})
        for i in range(pair_count):
            ne_entry = epi_ne.get(i, {})
            for direction in ("North", "East"):
                info = ne_entry.get(direction, {}) if isinstance(ne_entry, dict) else {}
                if bool(info.get("b16_disabled", False)):
                    continue
                if np.isfinite(float(info.get("b16", np.nan))):
                    include_b16 = True
                    break
            if include_b16:
                break
    else:
        epi = state.get("epistemic_b_by_epoch", {})
        for i in range(pair_count):
            info = epi.get(i, {}) if isinstance(epi, dict) else {}
            if bool(info.get("b16_disabled", False)):
                continue
            if np.isfinite(float(info.get("b16", np.nan))):
                include_b16 = True
                break
    case_keys = ["b50", "b84"] if not include_b16 else ["b16", "b50", "b84"]
    for case_key in case_keys:
        cfg_case = copy.deepcopy(config)
        cfg_case.prediction_coordinate_system = mode
        if mode == "N&E":
            models_case = _build_ne_models_for_case(state, pair_count, case_key)
            overrides_case = None
        else:
            models_case = state["models_by_epoch"]
            overrides_case = _build_fault_parallel_overrides_for_case(state, pair_count, case_key)
        pred = predict_pipe_motion_and_strain(
            config=cfg_case,
            pipe_axis=data["pipe_axis"],
            pipe_axis_segments=data.get("pipe_axis_segments"),
            fault_line=state["notebook_fault_ref"]["fault_line_truth"],
            left_area=state["notebook_fault_ref"]["left_area"],
            fault_polyline_for_projection=np.asarray(state["notebook_fault_ref"]["fl2"]),
            models_by_epoch=models_case,
            tangential_fit_by_epoch=state.get("tangential_fit_by_epoch", {}),
            normal_fit_by_epoch=state.get("normal_fit_by_epoch", {}),
            fit_overrides_by_epoch=overrides_case,
            pair_count=pair_count,
            intersection_point=state["fit_band"]["intersection"],
            compute_strain=include_strain,
        )
        pred["prediction_case"] = case_key
        pred["prediction_mode"] = mode
        cases[case_key] = pred
    return cases


def _compute_final_epistemic_sigma(max_e16: float, max_e50: float, max_e84: float) -> float:
    if not (np.isfinite(max_e16) and np.isfinite(max_e50) and np.isfinite(max_e84)):
        return np.nan
    if max_e16 <= 0 or max_e50 <= 0 or max_e84 <= 0:
        return np.nan
    t1 = np.log(max_e84 / max_e50)
    t2 = np.log(max_e50 / max_e16)
    return float(np.sqrt((t1 * t1 + t2 * t2) / 2.0))


def _fit_sample_count_summary(fit_mask: np.ndarray, signed_distance: np.ndarray) -> Dict[str, int]:
    m = np.asarray(fit_mask, dtype=bool).reshape(-1)
    n = np.asarray(signed_distance, dtype=float).reshape(-1)
    total = int(min(m.size, n.size))
    if total <= 0:
        return {"total_samples": 0, "fit_samples_ne": 0, "fit_samples_fault": 0}
    return {
        "total_samples": total,
        "fit_samples_ne": int(np.count_nonzero(m[:total])),
        "fit_samples_fault": int(np.count_nonzero(m[:total])),
    }


def _prepare_ne_fit_samples(
    config: AppConfig,
    signed_distance: np.ndarray,
    displacement_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = np.asarray(signed_distance, dtype=float).reshape(-1)
    disp = np.asarray(displacement_xy, dtype=float)
    if disp.ndim != 2 or disp.shape[1] < 2:
        return np.array([], dtype=float), np.empty((0, 2), dtype=float), {"raw_count": 0.0, "used_count": 0.0}

    n = int(min(x.size, disp.shape[0]))
    if n <= 0:
        return np.array([], dtype=float), np.empty((0, 2), dtype=float), {"raw_count": 0.0, "used_count": 0.0}

    x = x[:n]
    disp = disp[:n, :2]
    valid = np.isfinite(x) & np.isfinite(disp[:, 0]) & np.isfinite(disp[:, 1])
    x_raw = x[valid]
    disp_raw = disp[valid]
    raw_count = int(x_raw.size)
    if raw_count <= 0:
        return np.array([], dtype=float), np.empty((0, 2), dtype=float), {"raw_count": 0.0, "used_count": 0.0}

    q = np.nanquantile(x_raw, [0.05, 0.5, 0.95]) if raw_count > 1 else np.array([x_raw[0], x_raw[0], x_raw[0]])
    stats = {
        "raw_count": float(raw_count),
        "used_count": float(raw_count),
        "raw_x_min": float(np.nanmin(x_raw)),
        "raw_x_max": float(np.nanmax(x_raw)),
        "raw_x_q05": float(q[0]),
        "raw_x_q50": float(q[1]),
        "raw_x_q95": float(q[2]),
        "bin_width_m": 5.0,
        "occupied_bins": float(0),
        "fallback_raw": 0.0,
    }

    if str(getattr(config, "model_type", "sigmoid")).strip().lower() != "sigmoid":
        return x_raw, disp_raw, stats

    x_min = float(np.nanmin(x_raw))
    x_max = float(np.nanmax(x_raw))
    x_span = float(max(x_max - x_min, 0.0))
    if not np.isfinite(x_span) or x_span <= 1e-9:
        return x_raw, disp_raw, stats

    bin_width = float(stats["bin_width_m"])
    n_bins = int(max(np.ceil(x_span / max(bin_width, 1e-6)), 1))
    n_bins = int(np.clip(n_bins, 20, 400))
    edges = np.linspace(x_min, x_max, n_bins + 1, dtype=float)
    idx = np.searchsorted(edges, x_raw, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)

    xs = []
    ys = []
    for bi in range(n_bins):
        m = idx == bi
        if not np.any(m):
            continue
        x_bin = x_raw[m]
        y_bin = disp_raw[m]
        xs.append(float(np.nanmedian(x_bin)))
        ys.append(np.nanmedian(y_bin, axis=0))

    if not xs:
        return x_raw, disp_raw, stats

    x_use = np.asarray(xs, dtype=float)
    y_use = np.asarray(ys, dtype=float)
    min_req = _min_fit_points_required(config)
    if x_use.size < min_req:
        stats["fallback_raw"] = 1.0
        return x_raw, disp_raw, stats

    stats["occupied_bins"] = float(x_use.size)
    stats["used_count"] = float(x_use.size)
    return x_use, y_use, stats


def _max_abs_xy_diff_between_modes(cases_by_mode: Dict) -> np.ndarray:
    out = np.array([], dtype=float)
    if not isinstance(cases_by_mode, dict):
        return out
    ne = cases_by_mode.get("N&E", {})
    fp = cases_by_mode.get("Fault-parallel/perpendicular", {})
    pred_ne = ne.get("b50") if isinstance(ne, dict) else None
    pred_fp = fp.get("b50") if isinstance(fp, dict) else None
    if not (isinstance(pred_ne, dict) and isinstance(pred_fp, dict)):
        return out
    xy_ne = np.asarray(pred_ne.get("pipe_motion_xy_mm", []), dtype=float)
    xy_fp = np.asarray(pred_fp.get("pipe_motion_xy_mm", []), dtype=float)
    if xy_ne.ndim != 3 or xy_fp.ndim != 3:
        return out
    n_epoch = min(xy_ne.shape[0], xy_fp.shape[0])
    if n_epoch <= 0:
        return out
    for i in range(n_epoch):
        n_nodes = min(xy_ne.shape[1], xy_fp.shape[1])
        if n_nodes <= 0:
            continue
        diff = xy_ne[i, :n_nodes, :2] - xy_fp[i, :n_nodes, :2]
        mag = np.linalg.norm(diff, axis=1)
        out = np.append(out, float(np.nanmax(np.abs(mag))))
    return out


def _compute_segmentwise_s_nodes_offset(
    pipe_axis: np.ndarray,
    segment_ranges: list[tuple[int, int]],
    intersection_point: Optional[np.ndarray],
) -> np.ndarray:
    s_nodes_offset = np.zeros(pipe_axis.shape[0], dtype=float)
    if intersection_point is None:
        for s0, s1 in segment_ranges:
            seg = np.asarray(pipe_axis[s0:s1], dtype=float)
            if seg.shape[0] < 2:
                continue
            ds = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            ds = np.where(ds <= 1e-9, 1e-9, ds)
            s = np.concatenate([[0.0], np.cumsum(ds)])
            s_nodes_offset[s0:s1] = s
        return s_nodes_offset

    ipt = Point(float(intersection_point[0]), float(intersection_point[1]))
    for s0, s1 in segment_ranges:
        seg = np.asarray(pipe_axis[s0:s1], dtype=float)
        if seg.shape[0] < 2:
            continue
        ds = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        ds = np.where(ds <= 1e-9, 1e-9, ds)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_inter = LineString(seg).project(ipt)
        s_nodes_offset[s0:s1] = s - s_inter
    return s_nodes_offset


def predict_pipe_motion_and_strain(
    config: AppConfig,
    pipe_axis: np.ndarray,
    fault_line,
    left_area: Polygon,
    fault_polyline_for_projection: np.ndarray,
    models_by_epoch: Dict[int, object],
    pair_count: int,
    tangential_fit_by_epoch: Optional[Dict[int, object]] = None,
    normal_fit_by_epoch: Optional[Dict[int, object]] = None,
    fit_overrides_by_epoch: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    intersection_point: Optional[np.ndarray] = None,
    compute_strain: bool = True,
    pipe_axis_segments: Optional[list[np.ndarray]] = None,
):
    mesh_size = getattr(config, "mesh_size_m", 8.0)
    segments_in = []
    if isinstance(pipe_axis_segments, list) and pipe_axis_segments:
        for seg in pipe_axis_segments:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                segments_in.append(arr[:, :2])
    if not segments_in:
        axis_arr = np.asarray(pipe_axis, dtype=float)
        if axis_arr.ndim == 2 and axis_arr.shape[0] >= 2 and axis_arr.shape[1] >= 2:
            segments_in = [axis_arr[:, :2]]
    if not segments_in:
        raise ValueError("No valid pipeline geometry for displacement/strain prediction.")

    segment_nodes = []
    segment_ranges = []
    all_nodes = []
    cursor = 0
    for seg in segments_in:
        seg_resampled = _resample_polyline_by_spacing(np.asarray(seg, dtype=float), mesh_size)
        if seg_resampled.ndim != 2 or seg_resampled.shape[0] < 2:
            continue
        segment_nodes.append(seg_resampled[:, :2])
        all_nodes.append(seg_resampled[:, :2])
        n_seg = int(seg_resampled.shape[0])
        segment_ranges.append((cursor, cursor + n_seg))
        cursor += n_seg
    if not all_nodes:
        raise ValueError("No valid resampled pipeline geometry for prediction.")
    pipe_axis = np.vstack(all_nodes).astype(float)
    pipe_segment_ids = np.zeros(pipe_axis.shape[0], dtype=int)
    pipe_connectivity_parts = []
    for seg_id, (s0, s1) in enumerate(segment_ranges):
        pipe_segment_ids[s0:s1] = seg_id
        if s1 - s0 >= 2:
            conn = np.column_stack([np.arange(s0, s1 - 1), np.arange(s0 + 1, s1)]).astype(int)
            pipe_connectivity_parts.append(conn)
    pipe_connectivity = (
        np.vstack(pipe_connectivity_parts).astype(int)
        if pipe_connectivity_parts
        else np.empty((0, 2), dtype=int)
    )

    projection_polyline = np.asarray(fault_polyline_for_projection, dtype=float)
    if projection_polyline.ndim != 2 or projection_polyline.shape[0] < 2 or projection_polyline.shape[1] < 2:
        projection_polyline = np.asarray(fault_line.coords, dtype=float)
    s_nodes_offset = _compute_segmentwise_s_nodes_offset(pipe_axis, segment_ranges, intersection_point)
    # Signed distance by nearest segment to fault polyline (right+, left-).
    distance_pipe = compute_signed_distance_to_polyline(
        points=np.asarray(pipe_axis, dtype=float),
        polyline_coords=np.asarray(projection_polyline[:, :2], dtype=float),
        flip_sign=bool(getattr(config, "flip_signed_distance", False)),
    )
    step = max(int(config.bending_sample_step), 1)

    motions = np.zeros((pair_count, len(pipe_axis), 2), dtype=float)
    motions_fault = np.zeros((pair_count, len(pipe_axis), 2), dtype=float)
    soil_motions = np.zeros((pair_count, len(pipe_axis), 2), dtype=float)
    soil_motions_fault = np.zeros((pair_count, len(pipe_axis), 2), dtype=float)
    axial = np.zeros((pair_count, len(pipe_axis)), dtype=float)
    axial_smoothed = np.zeros((pair_count, len(pipe_axis)), dtype=float)
    n_nodes = len(pipe_axis)
    predict_mode = getattr(config, "prediction_coordinate_system", "N&E")
    pipe_tangents, pipe_normals = compute_unit_tangent_and_normal(
        np.asarray(pipe_axis, dtype=float), np.asarray(projection_polyline[:, :2], dtype=float)
    )
    ssi_enabled_cfg = bool(getattr(config, "ssi_enabled", False))
    ssi_active = bool(ssi_enabled_cfg and predict_mode == "Fault-parallel/perpendicular")
    ssi_params = _build_ssi_params_from_config(config) if ssi_active else None
    ssi_zone_epoch = []
    ssi_diag_epoch = []
    for i in range(pair_count):
        soil_tangent = np.zeros(n_nodes, dtype=float)
        soil_normal = np.zeros(n_nodes, dtype=float)
        if predict_mode == "Fault-parallel/perpendicular":
            fit_t = tangential_fit_by_epoch.get(i) if tangential_fit_by_epoch else None
            fit_n = normal_fit_by_epoch.get(i) if normal_fit_by_epoch else None
            if fit_t is None or fit_n is None:
                raise ValueError("Fault-parallel/perpendicular prediction requires tangential/normal fits from Step 2.")
            epoch_override = fit_overrides_by_epoch.get(i, {}) if fit_overrides_by_epoch else {}
            if config.model_type == "sigmoid":
                params_t = _resolve_fit_params_with_override(fit_t, epoch_override.get("Fault-parallel"))
                params_n = _resolve_fit_params_with_override(fit_n, epoch_override.get("Fault-perpendicular"))
                soil_tangent = sigmoid(distance_pipe, *params_t[:4])
                soil_normal = sigmoid(distance_pipe, *params_n[:4])
                zone_info = _sigmoid_transition_zone_from_b(
                    params_t[1] if len(params_t) > 1 else np.nan,
                    center_x=params_t[2] if len(params_t) > 2 else 0.0,
                )
            else:
                soil_tangent = _evaluate_shape_function(distance_pipe, fit_t, config.model_type)
                soil_normal = _evaluate_shape_function(distance_pipe, fit_n, config.model_type)
                zone_info = {"x01": np.nan, "x99": np.nan, "width": np.nan, "center": 0.0}
        else:
            model_ne = models_by_epoch[i]
            if config.model_type == "sigmoid":
                # Use directly fitted N/E shape functions for pipe displacement prediction.
                v_north = sigmoid(distance_pipe, *_normalize_sigmoid_params(model_ne["x_component"].params))
                v_east = sigmoid(distance_pipe, *_normalize_sigmoid_params(model_ne["y_component"].params))
                fit_t_ref = tangential_fit_by_epoch.get(i) if tangential_fit_by_epoch else None
                params_ref = (
                    _normalize_sigmoid_params(np.asarray(fit_t_ref.params, dtype=float))
                    if fit_t_ref is not None
                    else _normalize_sigmoid_params(np.asarray(model_ne["y_component"].params, dtype=float))
                )
                zone_info = _sigmoid_transition_zone_from_b(params_ref[1] if params_ref.size > 1 else np.nan, center_x=0.0)
            else:
                px = model_ne["x_component"].params  # North
                py = model_ne["y_component"].params  # East
                v_north = np.where(distance_pipe < 0, px[0] * distance_pipe + px[1], px[2] * distance_pipe + px[3])
                v_east = np.where(distance_pipe < 0, py[0] * distance_pipe + py[1], py[2] * distance_pipe + py[3])
                zone_info = {"x01": np.nan, "x99": np.nan, "width": np.nan, "center": 0.0}
            soil_ux = v_east
            soil_uy = v_north
            soil_tangent = soil_ux * pipe_tangents[:, 0] + soil_uy * pipe_tangents[:, 1]
            soil_normal = soil_ux * pipe_normals[:, 0] + soil_uy * pipe_normals[:, 1]

        soil_ux, soil_uy = compute_xy_displacement(pipe_normals, pipe_tangents, soil_normal, soil_tangent)
        soil_motions[i, :, 0] = soil_ux
        soil_motions[i, :, 1] = soil_uy
        soil_motions_fault[i, :, 0] = soil_tangent
        soil_motions_fault[i, :, 1] = soil_normal

        pipe_tangent = np.asarray(soil_tangent, dtype=float).copy()
        pipe_normal = np.asarray(soil_normal, dtype=float).copy()
        if ssi_active and isinstance(ssi_params, SSIParams):
            x01 = float(zone_info.get("x01", np.nan))
            x99 = float(zone_info.get("x99", np.nan))
            if np.isfinite(x01) and np.isfinite(x99):
                in_center = (distance_pipe >= x01) & (distance_pipe <= x99)
            else:
                in_center = np.zeros(distance_pipe.shape[0], dtype=bool)
            soil_fault = np.column_stack([soil_tangent, soil_normal])
            pf = np.asarray(soil_fault, dtype=float).copy()
            for s0, s1 in segment_ranges:
                if s1 - s0 <= 1:
                    continue
                seg_soil = soil_fault[s0:s1]
                seg_s = s_nodes_offset[s0:s1]
                seg_zone = in_center[s0:s1]
                seg_t = pipe_tangents[s0:s1]
                seg_n = pipe_normals[s0:s1]
                ssi_out = solve_pipe_ssi_fault_components(
                    soil_fault_mm=seg_soil,
                    s_nodes_offset_m=seg_s,
                    in_center_zone=seg_zone,
                    params=ssi_params,
                    tangent_xy=seg_t,
                    normal_xy=seg_n,
                )
                seg_pf = np.asarray(ssi_out.get("pipe_fault_mm", seg_soil), dtype=float)
                if seg_pf.ndim == 2 and seg_pf.shape[0] == (s1 - s0) and seg_pf.shape[1] >= 2:
                    pf[s0:s1, :2] = seg_pf[:, :2]
            pipe_tangent = pf[:, 0]
            pipe_normal = pf[:, 1]
            ssi_diag_epoch.append(
                {
                    "x01": float(x01),
                    "x99": float(x99),
                    "width_m": float(abs(x99 - x01)) if np.isfinite(x01) and np.isfinite(x99) else np.nan,
                    "center_m": float(zone_info.get("center", 0.0)),
                    "center_nodes": int(np.count_nonzero(in_center)),
                }
            )
            ssi_zone_epoch.append(in_center.astype(bool))
        else:
            ssi_diag_epoch.append(
                {
                    "x01": float(zone_info.get("x01", np.nan)),
                    "x99": float(zone_info.get("x99", np.nan)),
                    "width_m": float(zone_info.get("width", np.nan)),
                    "center_m": float(zone_info.get("center", 0.0)),
                    "center_nodes": 0,
                }
            )
            ssi_zone_epoch.append(np.zeros(distance_pipe.shape[0], dtype=bool))

        ux, uy = compute_xy_displacement(pipe_normals, pipe_tangents, pipe_normal, pipe_tangent)
        motions[i, :, 0] = ux
        motions[i, :, 1] = uy
        motions_fault[i, :, 0] = pipe_tangent
        motions_fault[i, :, 1] = pipe_normal

        if compute_strain:
            deformed = pipe_axis + motions[i] * 1e-3
            nodal = np.zeros(n_nodes, dtype=float)
            for s0, s1 in segment_ranges:
                if s1 - s0 < 2:
                    continue
                before_length = np.linalg.norm(np.diff(pipe_axis[s0:s1], axis=0), axis=1)
                after_length = np.linalg.norm(np.diff(deformed[s0:s1], axis=0), axis=1)
                diff_length = after_length - before_length
                safe_before = np.where(np.abs(before_length) < 1e-12, 1e-12, before_length)
                axial_seg = diff_length / safe_before
                avg = np.zeros((2, s1 - s0), dtype=float)
                avg[0, :-1] = axial_seg
                avg[1, 1:] = axial_seg
                nodal[s0:s1] = 0.5 * (avg[0] + avg[1])
            axial[i] = nodal

            raw_win = int(getattr(config, "axial_smoothing_window", 3))
            if raw_win <= 0:
                axial_smoothed[i] = nodal
            else:
                smoothed = nodal.copy()
                for s0, s1 in segment_ranges:
                    nodal_seg = nodal[s0:s1]
                    if nodal_seg.size < 9:
                        continue
                    win = max(raw_win, 9)
                    win = min(win, 101)
                    if win % 2 == 0:
                        win -= 1
                    n_lim = nodal_seg.size if nodal_seg.size % 2 == 1 else nodal_seg.size - 1
                    win = min(win, n_lim)
                    if win < 9:
                        continue
                    smoothed[s0:s1] = savgol_filter(nodal_seg, window_length=win, polyorder=7, mode="interp")
                axial_smoothed[i] = smoothed

    sample_indices = []
    for s0, s1 in segment_ranges:
        idx = np.arange(s0, s1, step, dtype=int)
        if idx.size == 0:
            idx = np.array([s0], dtype=int)
        if idx[-1] != s1 - 1:
            idx = np.append(idx, s1 - 1)
        sample_indices.append(idx)
    flat_sample_idx = np.concatenate(sample_indices) if sample_indices else np.array([], dtype=int)
    sampled_segment_ids = (
        np.concatenate([np.full(len(idx), i, dtype=int) for i, idx in enumerate(sample_indices)])
        if sample_indices
        else np.array([], dtype=int)
    )
    sampled_nodes = pipe_axis[flat_sample_idx] if flat_sample_idx.size > 0 else np.empty((0, 2), dtype=float)

    bending = np.zeros((max(pair_count, 1), sampled_nodes.shape[0]), dtype=float)
    if compute_strain:
        for i in range(pair_count):
            deformed = pipe_axis + motions[i] * 1e-3
            b_parts = []
            for idx in sample_indices:
                base_seg = pipe_axis[idx]
                def_seg = deformed[idx]
                b_parts.append(
                    compute_relative_bending_strain(
                        base_seg,
                        def_seg,
                        diameter_m=float(getattr(config, "ssi_D_out_m", 0.9144)),
                    )
                )
            bending[i] = np.concatenate(b_parts) if b_parts else np.array([], dtype=float)

    return {
        "distance_pipe": distance_pipe,
        "pipe_motion_xy_mm": motions,
        "pipe_motion_fault_mm": motions_fault,
        "soil_motion_xy_mm": soil_motions,
        "soil_motion_fault_mm": soil_motions_fault,
        "pipe_axis_nodes": pipe_axis,
        "pipe_axis_segments": [np.asarray(seg, dtype=float) for seg in segment_nodes],
        "pipe_connectivity": pipe_connectivity,
        "pipe_segment_ids": pipe_segment_ids,
        "axial_strain": axial,
        "axial_strain_smoothed": axial_smoothed,
        "bending_strain": bending,
        "pipe_axis_sampled": sampled_nodes,
        "pipe_axis_sampled_segment_ids": sampled_segment_ids,
        "s_nodes_offset": s_nodes_offset,
        "ssi_enabled": bool(ssi_active),
        "ssi_zone_by_epoch": np.asarray(ssi_zone_epoch, dtype=bool) if ssi_zone_epoch else np.empty((0, n_nodes), dtype=bool),
        "ssi_transition_by_epoch": ssi_diag_epoch,
        "prediction_coordinate_system": predict_mode,
    }


def _emit(progress, pct: int, msg: str):
    if progress:
        progress(pct, msg)


def _min_fit_points_required(config: AppConfig) -> int:
    model_type = str(getattr(config, "model_type", "sigmoid")).strip().lower()
    return 10 if model_type == "sigmoid" else 6


def _fit_band_params_changed(config: AppConfig, state: Dict) -> bool:
    prev = state.get("config")
    if isinstance(prev, AppConfig):
        prev_half = float(getattr(prev, "transect_half_length_m", np.nan))
        prev_off = float(getattr(prev, "fit_band_offset_m", np.nan))
    elif isinstance(prev, dict):
        prev_half = float(prev.get("transect_half_length_m", np.nan))
        prev_off = float(prev.get("fit_band_offset_m", np.nan))
    else:
        return False
    cur_half = float(getattr(config, "transect_half_length_m", np.nan))
    cur_off = float(getattr(config, "fit_band_offset_m", np.nan))
    return (not np.isfinite(prev_half)) or (not np.isfinite(prev_off)) or abs(prev_half - cur_half) > 1e-9 or abs(prev_off - cur_off) > 1e-9


def _fit_subset_counts(state: Dict) -> tuple[int, int]:
    signed = np.asarray(state.get("signed_distance", []), dtype=float).reshape(-1)
    mask = np.asarray(state.get("fit_mask", []), dtype=bool).reshape(-1)
    total = int(min(signed.size, mask.size))
    if total <= 0:
        return 0, 0
    return int(np.count_nonzero(mask[:total])), total


def _enforce_fit_subset_points(config: AppConfig, selected_count: int, total_count: int):
    min_req = _min_fit_points_required(config)
    if selected_count < min_req:
        raise ValueError(
            "Insufficient fit points in fitting band: "
            f"selected={selected_count}, total={total_count}, required>={min_req}. "
            "Increase Transect Half Length or Perpendicular Offset Lines, then run Step 2 again."
        )


def _recompute_fit_subset_for_step2(config: AppConfig, state: Dict) -> Dict[str, int]:
    needed = ["matched", "data", "notebook_fault_ref", "fit_band", "signed_distance"]
    if any(k not in state for k in needed):
        raise ValueError("Step 2 band recompute requires Step 1 outputs. Please run Step 1 first.")
    matched_coords = np.asarray(state["matched"].get("matched_coords", []), dtype=float)
    if matched_coords.ndim != 2 or matched_coords.shape[0] == 0:
        raise ValueError("No matched InSAR points available for Step 2 fitting.")
    fault_line = state["notebook_fault_ref"].get("fault_line_truth")
    pipe_axis = np.asarray(
        state["data"].get("pipe_axis_for_band", state["data"].get("pipe_axis", [])),
        dtype=float,
    )
    inter = np.asarray(state.get("fit_band", {}).get("intersection", []), dtype=float).reshape(-1)
    intersection_override = inter[:2] if inter.size >= 2 else None
    band = _compute_fit_band_mask(
        config=config,
        points=matched_coords,
        fault_line=fault_line,
        pipe_axis=pipe_axis,
        intersection_override=intersection_override,
    )
    signed_distance = np.asarray(state.get("signed_distance", []), dtype=float).reshape(-1)
    mask = np.asarray(band["mask"], dtype=bool).reshape(-1)
    total = int(min(mask.size, signed_distance.size))
    if total <= 0:
        raise ValueError("No valid points available for fitting.")
    mask = mask[:total]
    fit_distances = signed_distance[:total][mask]
    selected = int(fit_distances.size)
    _enforce_fit_subset_points(config, selected, total)
    state["fit_band"] = band
    state["fit_mask"] = mask
    state["fit_indices"] = np.where(mask)[0]
    state["fit_distances"] = fit_distances
    return {"selected": selected, "total": total}


def _sigmoid4(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return a * expit(b * (x - c)) + d


def _build_sliding_window_polygon(
    center_xy: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    offset_m: float,
    half_length_m: float,
    half_width_m: float,
) -> tuple[np.ndarray, Polygon]:
    center = np.asarray(center_xy, dtype=float).reshape(2) + float(offset_m) * np.asarray(tangent, dtype=float).reshape(2)
    n = np.asarray(normal, dtype=float).reshape(2)
    t = np.asarray(tangent, dtype=float).reshape(2)
    p1 = center - half_length_m * n + half_width_m * t
    p2 = center + half_length_m * n + half_width_m * t
    p3 = center + half_length_m * n - half_width_m * t
    p4 = center - half_length_m * n - half_width_m * t
    corners = np.vstack([p1, p2, p3, p4, p1])
    return corners, Polygon([p1, p2, p3, p4])


def _window_point_indices(poly: Polygon, coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return np.array([], dtype=int)
    idx = []
    for i in range(pts.shape[0]):
        p = Point(float(pts[i, 0]), float(pts[i, 1]))
        if poly.covers(p):
            idx.append(i)
    return np.asarray(idx, dtype=int)


def _fit_sigmoid4_window(x_data: np.ndarray, y_data: np.ndarray, min_points: int) -> Dict:
    x = np.asarray(x_data, dtype=float).reshape(-1)
    y = np.asarray(y_data, dtype=float).reshape(-1)
    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]
    if n < int(min_points):
        return {"success": False, "message": f"too_few_points(<{int(min_points)})"}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size < int(min_points):
            return {"success": False, "message": "too_few_finite_points"}
    try:
        p0_b = max(abs(float(np.nanstd(y))) / max(float(np.nanstd(x)), 1e-6), 0.1)
        p0 = [float(np.nanmax(y) - np.nanmin(y)), float(max(p0_b, 1e-3)), 0.0, float(np.nanmin(y))]
        popt, pcov = curve_fit(
            _sigmoid4,
            x,
            y,
            p0=p0,
            bounds=([1e-12, 1e-8, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=150000,
        )
        y_fit = _sigmoid4(x, *popt)
        rmse = float(np.sqrt(np.mean((y - y_fit) ** 2)))
        std = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(4, np.nan, dtype=float)
        b_at_lower_bound = bool(np.isfinite(popt[1]) and abs(float(popt[1]) - 1e-8) <= 1e-10)
        return {
            "success": True,
            "message": "ok",
            "a": float(popt[0]),
            "b": float(popt[1]),
            "c": float(popt[2]),
            "d": float(popt[3]),
            "std_a": float(std[0]) if np.isfinite(std[0]) else np.nan,
            "std_b": float(std[1]) if np.isfinite(std[1]) else np.nan,
            "std_c": float(std[2]) if np.isfinite(std[2]) else np.nan,
            "std_d": float(std[3]) if np.isfinite(std[3]) else np.nan,
            "rmse": rmse,
            "b_at_lower_bound": b_at_lower_bound,
        }
    except Exception as exc:
        return {"success": False, "message": f"fit_failed:{type(exc).__name__}"}


def _pielou_evenness_from_counts(counts: np.ndarray, n_bins: int) -> float:
    c = np.asarray(counts, dtype=float).reshape(-1)
    s = float(np.nansum(c))
    if s <= 0:
        return np.nan
    p = c / s
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    h = -float(np.sum(p * np.log(p)))
    return h / np.log(max(int(n_bins), 2))


def _attach_local_metrics(
    row: Dict,
    x_window: np.ndarray,
    y_window: np.ndarray,
    x_half: float,
    n_bins: int,
) -> Dict:
    x = np.asarray(x_window, dtype=float).reshape(-1)
    y = np.asarray(y_window, dtype=float).reshape(-1)
    n = int(min(x.size, y.size))
    x = x[:n]
    y = y[:n]
    row["n_local"] = 0
    row["uniform_expectation"] = np.nan
    row["bin_count_var"] = np.nan
    row["empty_bin_count"] = np.nan
    row["rmse_local"] = np.nan
    row["resid_var"] = np.nan
    row["resid_var_over_a"] = np.nan
    row["s1_balance"] = np.nan
    row["pielou_J"] = np.nan
    row["rel_std_b"] = np.nan
    row["jump_inside"] = 0
    row["n_transition"] = 0
    row["rmse_transition"] = np.nan
    row["a_rel_dev"] = np.nan
    row["_bin_centers"] = np.array([], dtype=float)
    row["_bin_counts"] = np.array([], dtype=int)
    if n <= 0:
        return row
    c = float(row.get("c", 0.0))
    b = float(row.get("b", np.nan))
    std_b = float(row.get("std_b", np.nan))
    row["rel_std_b"] = (
        float(std_b / max(abs(b), 1e-12)) if np.isfinite(std_b) and np.isfinite(b) else np.nan
    )
    if np.isfinite(c) and np.isfinite(b) and abs(b) > 1e-12:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        row["jump_inside"] = int(xmin <= float(c) <= xmax)
        half_w = float(np.log(9.0) / abs(b))
        in_tr = (x >= (float(c) - half_w)) & (x <= (float(c) + half_w))
        x_tr = x[in_tr]
        y_tr = y[in_tr]
        row["n_transition"] = int(x_tr.size)
        if x_tr.size > 0:
            y_hat_tr = _sigmoid4(x_tr, float(row.get("a", np.nan)), b, float(c), float(row.get("d", np.nan)))
            row["rmse_transition"] = float(np.sqrt(np.mean((y_tr - y_hat_tr) ** 2)))
    lo = c - float(x_half)
    hi = c + float(x_half)
    in_local = (x >= lo) & (x <= hi)
    x_local = x[in_local]
    y_local = y[in_local]
    row["n_local"] = int(x_local.size)
    if x_local.size <= 0:
        return row
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    counts, _ = np.histogram(x_local, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    row["uniform_expectation"] = float(np.mean(counts)) if counts.size > 0 else np.nan
    row["bin_count_var"] = float(np.var(counts, ddof=1)) if counts.size > 1 else 0.0
    row["empty_bin_count"] = int(np.sum(counts == 0))
    y_hat = _sigmoid4(x_local, float(row["a"]), float(row["b"]), float(row["c"]), float(row["d"]))
    resid = y_local - y_hat
    row["rmse_local"] = float(np.sqrt(np.mean(resid ** 2))) if resid.size > 0 else np.nan
    row["resid_var"] = float(np.var(resid, ddof=1)) if resid.size > 1 else 0.0
    a = float(row.get("a", np.nan))
    row["resid_var_over_a"] = float(row["resid_var"] / a) if np.isfinite(a) and abs(a) > 1e-12 else np.nan
    n_left = int(np.sum((x_local >= lo) & (x_local < c)))
    n_right = int(np.sum((x_local >= c) & (x_local <= hi)))
    denom = n_left + n_right
    row["s1_balance"] = float(1.0 - abs(n_left - n_right) / denom) if denom > 0 else np.nan
    row["pielou_J"] = _pielou_evenness_from_counts(counts, n_bins=int(n_bins))
    row["_bin_centers"] = centers
    row["_bin_counts"] = counts
    return row


def _detect_b_outliers_findpeaks(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()
    out["is_outlier"] = False
    out["outlier_score"] = np.nan
    out["outlier_reason"] = ""
    if out.empty or not bool(params.get("enabled", True)):
        return out
    valid = out["success"].astype(bool) & np.isfinite(out["b"].to_numpy(dtype=float))
    if int(np.count_nonzero(valid)) < 3:
        return out
    sub = out.loc[valid].copy().sort_values("offset").reset_index()
    b = sub["b"].to_numpy(dtype=float)
    prom = float(params.get("prominence", 0.2))
    dist = max(int(params.get("distance", 2)), 1)
    h_val = params.get("height", None)
    height = None if h_val is None else float(h_val)
    p_hi, _ = find_peaks(b, prominence=prom, distance=dist, height=height)
    p_lo, _ = find_peaks(-b, prominence=prom, distance=dist, height=height)
    peaks_idx = set([int(i) for i in p_hi] + [int(i) for i in p_lo])
    z_th = float(params.get("zscore_threshold", 3.0))
    med = float(np.median(b))
    mad = float(np.median(np.abs(b - med)))
    robust_scale = 1.4826 * mad if mad > 1e-12 else float(np.std(b))
    robust_scale = robust_scale if robust_scale > 1e-12 else 1.0
    z = np.abs((b - med) / robust_scale)
    z_idx = set(np.where(z >= z_th)[0].astype(int).tolist())
    all_idx = sorted(peaks_idx | z_idx)
    for k in all_idx:
        raw_idx = int(sub.loc[k, "index"])
        reasons = []
        if k in peaks_idx:
            reasons.append("peak")
        if k in z_idx:
            reasons.append("zscore")
        out.loc[raw_idx, "is_outlier"] = True
        out.loc[raw_idx, "outlier_score"] = float(z[k])
        out.loc[raw_idx, "outlier_reason"] = "+".join(reasons)
    return out


def _weighted_median(values, weights) -> float:
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = int(min(v.size, w.size))
    v = v[:n]
    w = w[:n]
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[keep]
    w = w[keep]
    if v.size == 0:
        return np.nan
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0:
        return np.nan
    cdf = cdf / total
    return float(v[int(np.searchsorted(cdf, 0.5, side="left"))])


AGGREGATION_METRICS_SMALLER_BETTER = (
    "bin_count_var",
    "empty_bin_count",
    "rmse_local",
    "resid_var_over_a",
    "rmse_transition",
    "rel_std_b",
    "a_rel_dev",
)
AGGREGATION_METRICS_LARGER_BETTER = (
    "s1_balance",
    "pielou_J",
    "n_transition",
    "jump_inside",
)
AGGREGATION_METRIC_ORDER = (
    "bin_count_var",
    "empty_bin_count",
    "rmse_local",
    "resid_var_over_a",
    "s1_balance",
    "pielou_J",
    "n_transition",
    "rmse_transition",
    "rel_std_b",
    "a_rel_dev",
    "jump_inside",
)


def default_aggregation_metric_weights() -> Dict[str, float]:
    return {name: 1.0 for name in AGGREGATION_METRIC_ORDER}


def _normalize_metric_weights(metric_weights: Optional[Dict]) -> Dict[str, float]:
    known = set(AGGREGATION_METRIC_ORDER)
    if metric_weights is None or not isinstance(metric_weights, dict):
        return default_aggregation_metric_weights()
    out: Dict[str, float] = {}
    for raw_name, raw_w in metric_weights.items():
        name = str(raw_name)
        if name not in known:
            continue
        try:
            w = float(raw_w)
        except (TypeError, ValueError):
            continue
        if np.isfinite(w) and w > 0:
            out[name] = w
    return out


def _minmax01(x) -> np.ndarray:
    xx = np.asarray(x, dtype=float).reshape(-1)
    finite = xx[np.isfinite(xx)]
    if finite.size == 0:
        return np.full(xx.shape, 0.5, dtype=float)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if abs(hi - lo) < 1e-12:
        return np.full(xx.shape, 0.5, dtype=float)
    return (xx - lo) / (hi - lo)


def _run_dual_aggregation(
    df: pd.DataFrame,
    top_percent: float = 10.0,
    n_min_transition: int = 8,
    metric_weights: Optional[Dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    weights = _normalize_metric_weights(metric_weights)
    empty_summary = {
        "n_rows_for_aggregation": 0,
        "top_percent_for_aggregation": float(np.clip(float(top_percent), 0.1, 100.0)),
        "n_min_transition": int(n_min_transition),
        "rank_top10_mean_b": np.nan,
        "score_top10_mean_b": np.nan,
        "rank_head_mean_b": np.nan,
        "score_head_mean_b": np.nan,
        "top10_jaccard": np.nan,
        "top10_mean_b_abs_diff": np.nan,
        "b_recommended": np.nan,
        "aggregation_mode": "weighted_metric_consensus",
        "aggregation_metric_weights": dict(weights),
    }
    if not isinstance(df, pd.DataFrame) or df.empty or not weights:
        return pd.DataFrame(), pd.DataFrame(), empty_summary
    active = [(name, weights[name]) for name in AGGREGATION_METRIC_ORDER if name in weights and name in df.columns]
    if not active:
        return pd.DataFrame(), pd.DataFrame(), empty_summary

    base = df.copy()
    if "success" in base.columns:
        base = base.loc[base["success"].astype(bool)].copy()
    if "is_outlier" in base.columns:
        base = base.loc[~base["is_outlier"].fillna(False).astype(bool)].copy()
    active_names = [name for name, _ in active]
    if "n_transition" in active_names and "n_transition" in base.columns:
        base = base.loc[base["n_transition"].fillna(0).astype(float) >= float(n_min_transition)].copy()
    if "b" not in base.columns:
        return pd.DataFrame(), pd.DataFrame(), empty_summary
    needed = ["b"] + active_names
    base = base.dropna(subset=[c for c in needed if c in base.columns]).copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), empty_summary
    finite_b = np.isfinite(base["b"].to_numpy(dtype=float))
    for name in active_names:
        finite_b = finite_b & np.isfinite(base[name].to_numpy(dtype=float))
    base = base.loc[finite_b].copy()
    empty_summary["n_rows_for_aggregation"] = int(len(base))
    if len(base) < 3:
        return pd.DataFrame(), pd.DataFrame(), empty_summary

    smaller = set(AGGREGATION_METRICS_SMALLER_BETTER)
    rank_df = base.copy()
    rank_sum = np.zeros(len(rank_df), dtype=float)
    for name, w in active:
        ascending = name in smaller
        ranks = rank_df[name].rank(ascending=ascending, method="average").to_numpy(dtype=float)
        rank_df[f"rank_{name}"] = ranks
        rank_sum = rank_sum + float(w) * ranks
    rank_df["rank_sum"] = rank_sum
    eps = 1e-12
    w_a = np.where(np.isfinite(rank_sum) & (rank_sum > 0), 1.0 / np.maximum(rank_sum, eps), 0.0)
    rank_df["weight_a"] = w_a
    rank_df["final_rank"] = rank_df["rank_sum"].rank(ascending=True, method="average")
    rank_df = rank_df.sort_values("final_rank").reset_index(drop=True)

    score_df = base.copy()
    score_acc = np.zeros(len(score_df), dtype=float)
    w_sum = 0.0
    for name, w in active:
        raw = score_df[name].to_numpy(dtype=float)
        norm = _minmax01(-raw if name in smaller else raw)
        score_df[f"norm_{name}"] = norm
        score_acc = score_acc + float(w) * norm
        w_sum += float(w)
    score_mean = score_acc / w_sum if w_sum > 0 else score_acc
    score_df["score_mean"] = score_mean
    score_df["score_rank"] = score_df["score_mean"].rank(ascending=False, method="average")
    score_df = score_df.sort_values("score_mean", ascending=False).reset_index(drop=True)

    bvals = base["b"].to_numpy(dtype=float)
    rank_cons = _weighted_median(bvals, w_a)
    score_cons = _weighted_median(bvals, score_mean)
    head = float(np.nanmedian(bvals)) if np.any(np.isfinite(bvals)) else np.nan
    rec = np.nan
    if np.isfinite(rank_cons) and np.isfinite(score_cons):
        rec = float(0.5 * (rank_cons + score_cons))
    elif np.isfinite(rank_cons):
        rec = rank_cons
    elif np.isfinite(score_cons):
        rec = score_cons
    w_med = float(np.nanmedian(w_a[w_a > 0])) if np.any(w_a > 0) else np.nan
    q_med = float(np.nanmedian(score_mean[np.isfinite(score_mean)])) if np.any(np.isfinite(score_mean)) else np.nan
    a_set = set(rank_df.loc[rank_df["weight_a"] >= w_med, "offset"].astype(float).to_list()) if np.isfinite(w_med) and "offset" in rank_df.columns else set()
    b_set = set(score_df.loc[score_df["score_mean"] >= q_med, "offset"].astype(float).to_list()) if np.isfinite(q_med) and "offset" in score_df.columns else set()
    jac = float(len(a_set & b_set) / len(a_set | b_set)) if len(a_set | b_set) > 0 else np.nan
    summary = {
        "n_rows_for_aggregation": int(len(base)),
        "top_percent_for_aggregation": float(np.clip(float(top_percent), 0.1, 100.0)),
        "n_min_transition": int(n_min_transition),
        "rank_top10_mean_b": rank_cons,
        "score_top10_mean_b": score_cons,
        "rank_head_mean_b": head,
        "score_head_mean_b": head,
        "top10_jaccard": jac,
        "top10_mean_b_abs_diff": (
            float(abs(rank_cons - score_cons))
            if np.isfinite(rank_cons) and np.isfinite(score_cons)
            else np.nan
        ),
        "b_recommended": rec,
        "aggregation_mode": "weighted_metric_consensus",
        "aggregation_metric_weights": {name: float(w) for name, w in active},
    }
    return rank_df, score_df, summary


def run_step_sigmoid_b_enhancement(
    config: AppConfig,
    state: Dict,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    required = ["NEZ", "signed_distance", "fault_tangents", "fault_normals", "matched", "fit_band", "data", "notebook_fault_ref"]
    if any(k not in state for k in required):
        raise ValueError("Sigmoid b enhancement requires Step 1/2 results. Run Step 1 and Step 2 first.")
    req = state.get("b_enhancement_request", {}) if isinstance(state.get("b_enhancement_request", {}), dict) else {}
    epoch_idx = int(req.get("epoch_idx", 0))
    nez = np.asarray(state["NEZ"], dtype=float)
    if nez.ndim != 3 or nez.shape[0] <= 0:
        raise ValueError("Invalid NEZ shape for enhancement.")
    epoch_idx = int(np.clip(epoch_idx, 0, nez.shape[0] - 1))
    params = {
        "offset_min_m": float(req.get("offset_min_m", -300.0)),
        "offset_max_m": float(req.get("offset_max_m", 400.0)),
        "offset_step_m": float(req.get("offset_step_m", 10.0)),
        "window_half_length_m": float(req.get("window_half_length_m", 600.0)),
        "window_half_width_m": float(req.get("window_half_width_m", 120.0)),
        "x_half_for_c_window": float(req.get("x_half_for_c_window", 100.0)),
        "min_points_per_window": int(req.get("min_points_per_window", 30)),
        "n_bins": int(req.get("n_bins", 20)),
        "remove_outliers_for_aggregation": bool(req.get("remove_outliers_for_aggregation", True)),
        "n_min_transition": int(req.get("n_min_transition", 8)),
        "peak_detection": {
            "enabled": bool(req.get("peak_enabled", True)),
            "prominence": float(req.get("peak_prominence", 0.2)),
            "distance": int(req.get("peak_distance", 2)),
            "height": None if req.get("peak_height", None) in (None, "", "None") else float(req.get("peak_height")),
            "zscore_threshold": float(req.get("peak_zscore_threshold", 3.0)),
        },
    }
    if params["offset_step_m"] <= 0:
        raise ValueError("offset_step_m must be > 0.")
    offsets = np.arange(
        params["offset_min_m"],
        params["offset_max_m"] + 0.5 * params["offset_step_m"],
        params["offset_step_m"],
    )
    _emit(progress, 8, f"B enhancement setup done. epoch={epoch_idx + 1}, windows={len(offsets)}")

    coords = np.asarray(state["matched"].get("matched_coords", []), dtype=float)
    signed_distance = np.asarray(state["signed_distance"], dtype=float).reshape(-1)
    tangents = np.asarray(state["fault_tangents"], dtype=float)
    normals = np.asarray(state["fault_normals"], dtype=float)
    n_pts = int(min(coords.shape[0], signed_distance.size, tangents.shape[0], normals.shape[0], nez.shape[1]))
    if n_pts <= 0:
        raise ValueError("Enhancement input arrays are empty.")
    coords = coords[:n_pts, :2]
    signed_distance = signed_distance[:n_pts]
    tangents = tangents[:n_pts, :2]
    normals = normals[:n_pts, :2]
    disp_swap = np.column_stack((nez[epoch_idx, :n_pts, 1], nez[epoch_idx, :n_pts, 0]))
    v_tangent = np.sum(disp_swap * tangents, axis=1)

    fault_ref = state.get("notebook_fault_ref", {})
    trace_type = str(fault_ref.get("fault_trace_type", getattr(config, "fault_trace_type", "selected_geometry")))
    if trace_type in {"linear_fit", "quadratic_fit"}:
        fault_line = fault_ref.get("fault_line_fitted", fault_ref.get("fault_line_truth"))
    else:
        fault_line = fault_ref.get("fault_line_selected", fault_ref.get("fault_line_truth"))
    if not isinstance(fault_line, LineString):
        fault_line = fault_ref.get("fault_line_truth")
    if not isinstance(fault_line, LineString):
        raise ValueError("No valid fault line found for b enhancement.")

    inter = np.asarray(state.get("fit_band", {}).get("intersection", []), dtype=float).reshape(-1)
    if inter.size < 2 or not np.isfinite(inter[:2]).all():
        pipe_axis = np.asarray(state.get("data", {}).get("pipe_axis_for_band", state.get("data", {}).get("pipe_axis", [])), dtype=float)
        if pipe_axis.ndim != 2 or pipe_axis.shape[0] < 2:
            raise ValueError("No valid pipe axis for enhancement intersection.")
        inter = _pick_intersection_point(LineString(pipe_axis[:, :2]), fault_line)
    inter = inter[:2]
    fault_coords = np.asarray(fault_line.coords, dtype=float)
    seg_vec = np.diff(fault_coords[:, :2], axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    proj = float(fault_line.project(Point(float(inter[0]), float(inter[1]))))
    cum = np.insert(np.cumsum(seg_len), 0, 0.0)
    idx_fault = int(np.clip(np.searchsorted(cum, proj, side="right") - 1, 0, len(seg_len) - 1))
    if seg_len[idx_fault] <= 1e-12:
        nz = np.where(seg_len > 1e-12)[0]
        idx_fault = int(nz[0]) if nz.size > 0 else 0
    tangent = seg_vec[idx_fault]
    tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 1e-12 else np.array([1.0, 0.0], dtype=float)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)

    _emit(progress, 18, "Running sliding-window sigmoid fits")
    rows = []
    n_offsets = max(int(len(offsets)), 1)
    for i, s in enumerate(offsets):
        corners, poly = _build_sliding_window_polygon(
            center_xy=inter,
            tangent=tangent,
            normal=normal,
            offset_m=float(s),
            half_length_m=float(params["window_half_length_m"]),
            half_width_m=float(params["window_half_width_m"]),
        )
        idx_sel = _window_point_indices(poly, coords)
        row = {
            "offset": float(s),
            "success": False,
            "message": "",
            "n_points": int(idx_sel.size),
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "d": np.nan,
            "std_a": np.nan,
            "std_b": np.nan,
            "std_c": np.nan,
            "std_d": np.nan,
            "rmse": np.nan,
            "b_at_lower_bound": False,
            "n_local": 0,
            "uniform_expectation": np.nan,
            "bin_count_var": np.nan,
            "empty_bin_count": np.nan,
            "rmse_local": np.nan,
            "resid_var": np.nan,
            "resid_var_over_a": np.nan,
            "s1_balance": np.nan,
            "pielou_J": np.nan,
            "rel_std_b": np.nan,
            "jump_inside": 0,
            "n_transition": 0,
            "rmse_transition": np.nan,
            "a_rel_dev": np.nan,
            "_x_window": np.array([], dtype=float),
            "_y_window": np.array([], dtype=float),
            "_bin_centers": np.array([], dtype=float),
            "_bin_counts": np.array([], dtype=int),
            "_window_corners": corners,
        }
        if idx_sel.size < int(params["min_points_per_window"]):
            row["message"] = f"too_few_points(<{int(params['min_points_per_window'])})"
            rows.append(row)
            continue
        x_window = signed_distance[idx_sel]
        y_window = v_tangent[idx_sel]
        row["_x_window"] = np.asarray(x_window, dtype=float)
        row["_y_window"] = np.asarray(y_window, dtype=float)
        fit = _fit_sigmoid4_window(x_window, y_window, int(params["min_points_per_window"]))
        row["success"] = bool(fit.get("success", False))
        row["message"] = str(fit.get("message", ""))
        if row["success"]:
            for key in ("a", "b", "c", "d", "std_a", "std_b", "std_c", "std_d", "rmse"):
                row[key] = float(fit.get(key, np.nan))
            row["b_at_lower_bound"] = bool(fit.get("b_at_lower_bound", False))
            _attach_local_metrics(
                row,
                np.asarray(x_window, dtype=float),
                np.asarray(y_window, dtype=float),
                x_half=float(params["x_half_for_c_window"]),
                n_bins=int(params["n_bins"]),
            )
        rows.append(row)
        if i % 5 == 0 or i == n_offsets - 1:
            pct = int(18 + 40 * (i + 1) / n_offsets)
            _emit(progress, pct, f"Sliding fits {i + 1}/{n_offsets}")
    fit_df = pd.DataFrame(rows).sort_values("offset").reset_index(drop=True)
    _emit(progress, 63, "Running peak-based outlier detection")
    fit_df = _detect_b_outliers_findpeaks(fit_df, params["peak_detection"])
    if "a_rel_dev" not in fit_df.columns:
        fit_df["a_rel_dev"] = np.nan
    success_a = fit_df["success"].astype(bool) & np.isfinite(pd.to_numeric(fit_df["a"], errors="coerce").to_numpy(dtype=float))
    if int(np.count_nonzero(success_a.to_numpy())) > 0:
        med_a = float(np.nanmedian(pd.to_numeric(fit_df.loc[success_a, "a"], errors="coerce").to_numpy(dtype=float)))
        denom_a = max(abs(med_a), 1e-12)
        a_vals = pd.to_numeric(fit_df["a"], errors="coerce").to_numpy(dtype=float)
        fit_df.loc[success_a, "a_rel_dev"] = np.abs(a_vals[success_a.to_numpy()] - med_a) / denom_a

    coverage_cols = [
        "offset",
        "success",
        "message",
        "a",
        "b",
        "c",
        "d",
        "b_at_lower_bound",
        "n_points",
        "n_local",
        "uniform_expectation",
        "bin_count_var",
        "empty_bin_count",
        "rmse_local",
        "resid_var",
        "resid_var_over_a",
        "rel_std_b",
        "jump_inside",
        "n_transition",
        "rmse_transition",
        "a_rel_dev",
        "_bin_centers",
        "_bin_counts",
        "is_outlier",
        "outlier_score",
        "outlier_reason",
    ]
    coverage_residual_df = fit_df[[c for c in coverage_cols if c in fit_df.columns]].copy()
    balance_cols = [
        "offset",
        "success",
        "message",
        "b",
        "c",
        "s1_balance",
        "pielou_J",
        "rel_std_b",
        "jump_inside",
        "n_transition",
        "rmse_transition",
        "a_rel_dev",
        "is_outlier",
        "outlier_score",
        "outlier_reason",
    ]
    balance_evenness_df = fit_df[[c for c in balance_cols if c in fit_df.columns]].copy()
    _emit(progress, 74, "Computing dual aggregation and recommendation")
    agg_cols = [
        "offset",
        "success",
        "b",
        "a",
        "is_outlier",
        "bin_count_var",
        "empty_bin_count",
        "rmse_local",
        "resid_var_over_a",
        "s1_balance",
        "pielou_J",
        "jump_inside",
        "n_transition",
        "rmse_transition",
        "rel_std_b",
        "a_rel_dev",
    ]
    merged = fit_df[[c for c in agg_cols if c in fit_df.columns]].copy()
    if bool(params["remove_outliers_for_aggregation"]) and "is_outlier" in merged.columns:
        merged = merged.loc[~merged["is_outlier"].fillna(False).astype(bool)].copy()
    top_pct = float(req.get("top_percent_for_aggregation", 10.0))
    metric_weights = req.get("aggregation_metric_weights", None)
    method_a_df, method_b_df, summary = _run_dual_aggregation(
        merged.copy(),
        top_percent=top_pct,
        n_min_transition=int(params["n_min_transition"]),
        metric_weights=metric_weights,
    )

    b_current = np.nan
    fit_t = state.get("tangential_fit_by_epoch", {}).get(epoch_idx)
    if fit_t is not None:
        params_t = np.asarray(getattr(fit_t, "params", []), dtype=float).reshape(-1)
        if params_t.size > 1 and np.isfinite(params_t[1]):
            b_current = float(params_t[1])
    summary["b_current_step2"] = b_current
    summary["epoch_index"] = int(epoch_idx)
    summary["offset_count"] = int(len(offsets))
    summary["success_window_count"] = int(np.count_nonzero(fit_df["success"].to_numpy(dtype=bool)))
    summary["outlier_count"] = int(np.count_nonzero(fit_df["is_outlier"].to_numpy(dtype=bool)))
    summary["b_at_lower_bound_count"] = int(np.count_nonzero(fit_df["b_at_lower_bound"].to_numpy(dtype=bool)))
    summary["computed_at"] = datetime.now().isoformat(timespec="seconds")

    map_pipe_axis = np.asarray(
        state.get("data", {}).get("pipe_axis_for_band", state.get("data", {}).get("pipe_axis", [])),
        dtype=float,
    )
    if map_pipe_axis.ndim != 2 or map_pipe_axis.shape[0] < 2:
        map_pipe_axis = np.empty((0, 2), dtype=float)
    map_fault_coords = np.asarray(fault_line.coords, dtype=float) if isinstance(fault_line, LineString) else np.empty((0, 2), dtype=float)
    ex_offsets = [0.0, float(params["offset_min_m"]), float(params["offset_max_m"])]
    ex_windows = {}
    if isinstance(fit_df, pd.DataFrame) and not fit_df.empty:
        offsets_arr = fit_df["offset"].to_numpy(dtype=float)
        for s_ref in ex_offsets:
            idx_near = int(np.argmin(np.abs(offsets_arr - float(s_ref))))
            rr = fit_df.iloc[idx_near]
            ex_windows[str(float(s_ref))] = np.asarray(
                rr.get("_window_corners", np.empty((0, 2), dtype=float)),
                dtype=float,
            )

    enh = {
        "epoch_idx": int(epoch_idx),
        "trace_type": str(trace_type),
        "intersection_xy": np.asarray(inter, dtype=float),
        "fault_tangent_xy": np.asarray(tangent, dtype=float),
        "fault_normal_xy": np.asarray(normal, dtype=float),
        "offsets": np.asarray(offsets, dtype=float),
        "params": params,
        "fit_df": fit_df,
        "coverage_residual_df": coverage_residual_df,
        "balance_evenness_df": balance_evenness_df,
        "merged_df": merged.reset_index(drop=True),
        "method_a_df": method_a_df,
        "method_b_df": method_b_df,
        "summary": summary,
        "map_overview": {
            "matched_coords": coords,
            "color_values": np.asarray(nez[epoch_idx, :n_pts, 0], dtype=float),
            "pipe_axis": map_pipe_axis,
            "fault_coords": map_fault_coords,
            "intersection_xy": np.asarray(inter, dtype=float),
            "example_windows": ex_windows,
        },
    }
    by_epoch = state.setdefault("b_enhancement_by_epoch", {})
    if isinstance(by_epoch, dict):
        by_epoch[int(epoch_idx)] = enh
    state["b_enhancement_last"] = enh
    _emit(progress, 100, "Sigmoid b enhancement completed")
    return state


def run_step_match_3d(
    config: AppConfig,
    state: Optional[Dict] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    state = state or {}
    data_key = _build_step1_data_key(config)
    with _STEP1_CACHE_LOCK:
        cached_data = _STEP1_DATA_CACHE.get(data_key)
    if cached_data is None:
        _emit(progress, 5, "Loading input data")
        loaded = load_inputs(config)
        with _STEP1_CACHE_LOCK:
            _STEP1_DATA_CACHE[data_key] = copy.deepcopy(loaded)
        data = loaded
        data_cache_hit = False
    else:
        _emit(progress, 5, "Loading input data (cache hit)")
        data = copy.deepcopy(cached_data)
        data_cache_hit = True

    _emit(progress, 12, "Preparing selected fault/pipeline geometry")
    selected_fault_idxs = list(state.get("selected_fault_idxs", []))
    selected_pipeline_idxs = list(state.get("selected_pipeline_idxs", []))
    pipe_segments_before_clip = _build_pipe_segments_from_selected(data, selected_pipeline_idxs)
    fault_line_default, fault_line_selected, fault_line_fitted_raw, fault_line_fitted_extended, fault_summary = _build_fault_trace_from_selected(
        config=config,
        fault_data=data["fault_data"],
        selected_fault_idxs=selected_fault_idxs,
    )
    effective_aspect_deg, aspect_source_used = _resolve_effective_constraint_aspect(
        config=config,
        fault_data=data["fault_data"],
        selected_fault_idxs=selected_fault_idxs,
        fallback_line=fault_line_default,
    )
    cfg_solve = replace(config, constraint_aspect_deg=float(effective_aspect_deg))
    trace_type = str(getattr(config, "fault_trace_type", "selected_geometry"))
    inter_info, fault_line_effective, inter_source = _resolve_intersection_center_with_trace_fallback(
        pipe_segments=pipe_segments_before_clip,
        trace_type=trace_type,
        fitted_raw=fault_line_fitted_raw,
        fitted_extended=fault_line_fitted_extended,
        default_trace=fault_line_default,
    )
    inter_center = np.asarray(inter_info["center_xy"], dtype=float)
    pipe_segments_clipped, clip_summary = _clip_pipe_segments_around_center(
        pipe_segments=pipe_segments_before_clip,
        center_xy=inter_center,
        half_length_m=float(getattr(config, "pipe_clip_half_length_m", 300.0)),
    )
    pipe_axis_for_band = _select_pipe_segment_near_center(pipe_segments_clipped, inter_center)
    data["pipe_axis_segments"] = [np.asarray(seg, dtype=float)[:, :2] for seg in pipe_segments_clipped]
    data["pipe_axis"] = _flatten_pipe_segments(data["pipe_axis_segments"])
    data["pipe_axis_for_band"] = pipe_axis_for_band
    data["fault_polyline"] = fault_line_effective

    match_key = _build_step1_match_key(config, data_key)
    with _STEP1_CACHE_LOCK:
        cached_match = _STEP1_MATCH_CACHE.get(match_key)
    if cached_match is None:
        _emit(progress, 20, "Temporal and spatial matching")
        matched = match_insar(config, data["asc_data"], data["desc_data"])
        with _STEP1_CACHE_LOCK:
            _STEP1_MATCH_CACHE[match_key] = copy.deepcopy(matched)
        match_cache_hit = False
    else:
        _emit(progress, 20, "Temporal and spatial matching (cache hit)")
        matched = copy.deepcopy(cached_match)
        match_cache_hit = True

    nez_key = _build_step1_nez_key(cfg_solve, match_key, effective_aspect_deg)
    with _STEP1_CACHE_LOCK:
        cached_sol = _STEP1_NEZ_CACHE.get(nez_key)
    if cached_sol is None:
        _emit(progress, 35, "Solving 3D displacement")
        sol = solve_3d_displacement(cfg_solve, matched)
        with _STEP1_CACHE_LOCK:
            _STEP1_NEZ_CACHE[nez_key] = copy.deepcopy(sol)
        nez_cache_hit = False
    else:
        _emit(progress, 35, "Solving 3D displacement (cache hit)")
        sol = copy.deepcopy(cached_sol)
        nez_cache_hit = True
    nez = sol["NEZ"]

    _emit(progress, 45, "Preparing fault reference and fitting band")
    t0 = time.perf_counter()
    distance_info = compute_fault_distance_field(config, matched["matched_coords"], fault_line_effective)
    t1 = time.perf_counter()
    _emit(progress, 55, f"Distance field ready ({len(matched['matched_coords'])} points, {max(len(fault_line_effective.coords)-1,0)} segments)")
    signed_distance = np.asarray(distance_info["signed_distance"], dtype=float)
    fault_coords = np.asarray(fault_line_effective.coords, dtype=float)
    nb_fault = {
        "fault_line_truth": fault_line_effective,
        "fault_line_truth2": fault_line_effective,
        "fl2": fault_coords,
        "distances_truth": signed_distance,
        "left_area": None,
        "truth_idx": -1,
        "connector_idx": -1,
        "fault_line_selected": fault_line_selected,
        "fault_line_fitted_raw": fault_line_fitted_raw,
        "fault_line_fitted_extended": fault_line_fitted_extended,
        "fault_line_fitted": fault_line_effective if trace_type in {"linear_fit", "quadratic_fit"} else fault_line_fitted_raw,
        "fault_trace_type": trace_type,
        "intersection_trace_source": inter_source,
    }
    band = _compute_fit_band_mask(
        config=config,
        points=matched["matched_coords"],
        fault_line=fault_line_effective,
        pipe_axis=pipe_axis_for_band,
        intersection_override=inter_center,
    )
    t2 = time.perf_counter()
    _emit(progress, 65, "Fitting band mask prepared")
    filtered_mask = band["mask"]
    filtered_idx = np.where(filtered_mask)[0]
    fit_distances = signed_distance[filtered_mask]
    tangents_m, normals_m = compute_unit_tangent_and_normal(matched["matched_coords"], fault_coords)
    t3 = time.perf_counter()
    _emit(progress, 78, "Fault local tangent/normal projection completed")
    auto_view = _compute_step1_auto_view(pipe_axis_for_band, inter_center)
    fault_trace_summary = {
        **fault_summary,
        "constraint_aspect_source_used": str(aspect_source_used),
        "constraint_aspect_deg_effective": float(effective_aspect_deg),
        "constraint_aspect_deg_manual": float(getattr(config, "constraint_aspect_deg", np.nan)),
        "step1_cache_data_hit": bool(data_cache_hit),
        "step1_cache_match_hit": bool(match_cache_hit),
        "step1_cache_nez_hit": bool(nez_cache_hit),
        "selected_pipeline_count": int(len(selected_pipeline_idxs)),
        "selected_pipeline_idxs": selected_pipeline_idxs,
        "intersection_source": inter_source,
        "intersection_count": int(inter_info["intersection_count"]),
        "intersection_center_method": str(inter_info.get("intersection_center_method", "")),
        "intersection_center_x_m": float(inter_center[0]),
        "intersection_center_y_m": float(inter_center[1]),
        "intersection_points_xy": inter_info["intersection_points_xy"],
        "pipe_clip_half_length_m": float(getattr(config, "pipe_clip_half_length_m", 300.0)),
        **clip_summary,
    }
    timing = {
        "distance_field_s": float(t1 - t0),
        "fit_band_mask_s": float(t2 - t1),
        "tangent_projection_s": float(t3 - t2),
        "postprocess_s": float(time.perf_counter() - t3),
    }
    fault_trace_summary["step1_timing_s"] = timing
    _emit(
        progress,
        90,
        "Step 1 finalizing outputs "
        f"(distance={timing['distance_field_s']:.2f}s, mask={timing['fit_band_mask_s']:.2f}s, tangent={timing['tangent_projection_s']:.2f}s)",
    )
    _emit(
        progress,
        92,
        f"Step1 cache/data={data_cache_hit}, match={match_cache_hit}, nez={nez_cache_hit}, "
        f"constraint_aspect={effective_aspect_deg:.3f} ({aspect_source_used})",
    )
    _emit(progress, 100, "Step 1 completed")
    return {
        "config": config,
        "data": data,
        "matched": matched,
        "solution": sol,
        "distance": distance_info,
        "notebook_fault_ref": nb_fault,
        "fit_band": band,
        "fit_indices": filtered_idx,
        "fit_mask": filtered_mask,
        "fit_distances": fit_distances,
        "signed_distance": signed_distance,
        "fault_coords_for_projection": fault_coords,
        "fault_tangents": tangents_m,
        "fault_normals": normals_m,
        "NEZ": nez,
        "fault_trace_summary": fault_trace_summary,
        "step1_auto_view": auto_view,
    }


def run_step_fit(
    config: AppConfig,
    state: Dict,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    required = ["NEZ", "signed_distance", "fit_mask", "fit_distances", "fault_tangents", "fault_normals"]
    if any(k not in state for k in required):
        raise ValueError("Step 2 requires Step 1 results. Run Step 1 first.")

    fit_band_recomputed = False
    if _fit_band_params_changed(config, state):
        _emit(progress, 48, "Step 2 detected updated fitting-band parameters; recomputing fit subset")
        counts = _recompute_fit_subset_for_step2(config, state)
        fit_band_recomputed = True
    else:
        selected_count, total_count = _fit_subset_counts(state)
        _enforce_fit_subset_points(config, selected_count, total_count)
        counts = {"selected": selected_count, "total": total_count}

    _emit(progress, 55, "Fitting shape functions")
    nez = state["NEZ"]
    signed_distance = state["signed_distance"]
    fit_mask = state["fit_mask"]
    fit_distances = state["fit_distances"]
    tangents_m = state["fault_tangents"]
    normals_m = state["fault_normals"]

    models_by_epoch = {}
    normal_fit_by_epoch = {}
    tangential_fit_by_epoch = {}
    tangential_displacement_by_epoch = {}
    normal_displacement_by_epoch = {}
    epistemic_b_by_epoch = {}
    epistemic_b_ne_by_epoch = {}
    ne_fit_sample_stats_by_epoch = {}
    b16_floor = float(getattr(config, "constraint_b16_floor", 0.1))
    ne_raw_counts = []
    ne_used_counts = []
    ne_bins_used = []
    for i in range(nez.shape[0]):
        disp_xy = np.column_stack((nez[i, :, 0], nez[i, :, 1]))
        x_ne_raw = np.asarray(signed_distance[fit_mask], dtype=float)
        y_ne_raw = np.asarray(disp_xy[fit_mask], dtype=float)
        x_ne_fit, y_ne_fit, ne_stats = _prepare_ne_fit_samples(config, x_ne_raw, y_ne_raw)
        ne_fit_sample_stats_by_epoch[i] = ne_stats
        ne_raw_counts.append(int(ne_stats.get("raw_count", 0.0)))
        ne_used_counts.append(int(ne_stats.get("used_count", 0.0)))
        ne_bins_used.append(int(ne_stats.get("occupied_bins", 0.0)))
        models_by_epoch[i] = fit_model(config, x_ne_fit, y_ne_fit)

        disp_swap = np.column_stack((nez[i, :, 1], nez[i, :, 0]))
        v_tangent = np.sum(disp_swap * tangents_m, axis=1)
        v_normal = np.sum(disp_swap * normals_m, axis=1)
        tangential_displacement_by_epoch[i] = v_tangent[fit_mask]
        normal_displacement_by_epoch[i] = v_normal[fit_mask]

        if config.model_type == "sigmoid":
            fit_north = models_by_epoch[i]["x_component"]
            fit_east = models_by_epoch[i]["y_component"]
            epistemic_b_ne_by_epoch[i] = {
                "North": _build_epistemic_b_entry(fit_north, b16_floor=b16_floor),
                "East": _build_epistemic_b_entry(fit_east, b16_floor=b16_floor),
            }
            normal_fit_by_epoch[i] = _fit_sigmoid(fit_distances, v_normal[fit_mask], config)
            tangential_fit_by_epoch[i] = _fit_sigmoid(fit_distances, v_tangent[fit_mask], config)
            epistemic_b_by_epoch[i] = _build_epistemic_b_entry(tangential_fit_by_epoch[i], b16_floor=b16_floor)
        else:
            normal_fit_by_epoch[i] = _fit_piecewise_linear(fit_distances, v_normal[fit_mask])
            tangential_fit_by_epoch[i] = _fit_piecewise_linear(fit_distances, v_tangent[fit_mask])

    ne_raw_med = int(np.nanmedian(ne_raw_counts)) if ne_raw_counts else 0
    ne_used_med = int(np.nanmedian(ne_used_counts)) if ne_used_counts else 0
    ne_bins_med = int(np.nanmedian(ne_bins_used)) if ne_bins_used else 0
    fit_counts = _fit_sample_count_summary(fit_mask, signed_distance)
    fit_counts.update(
        {
            "fit_samples_ne_raw": ne_raw_med,
            "fit_samples_ne": ne_used_med,
            "fit_bins_ne": ne_bins_med,
            "fit_samples_ne_agg_enabled": int(bool(ne_bins_med > 0)),
        }
    )

    state.update(
        {
            "models_by_epoch": models_by_epoch,
            "normal_fit_by_epoch": normal_fit_by_epoch,
            "tangential_fit_by_epoch": tangential_fit_by_epoch,
            "fault_tangential_displacement_by_epoch": tangential_displacement_by_epoch,
            "fault_normal_displacement_by_epoch": normal_displacement_by_epoch,
            "fault_fit_override_by_epoch": {},
            "epistemic_b_by_epoch": epistemic_b_by_epoch,
            "epistemic_b_ne_by_epoch": epistemic_b_ne_by_epoch,
            "model_type": str(getattr(config, "model_type", "sigmoid")),
            "fit_sample_count_by_mode": fit_counts,
            "ne_fit_sample_stats_by_epoch": ne_fit_sample_stats_by_epoch,
            "fit_subset_count_selected": int(counts.get("selected", 0)),
            "fit_subset_count_total": int(counts.get("total", 0)),
            "fit_band_recomputed_in_step2": bool(fit_band_recomputed),
        }
    )
    state["config"] = copy.deepcopy(config)
    _emit(progress, 100, "Step 2 completed")
    return state


def run_step_strain(
    config: AppConfig,
    state: Dict,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    required = ["models_by_epoch", "NEZ", "data", "matched", "signed_distance"]
    if any(k not in state for k in required):
        raise ValueError("Step 4 requires Step 2 results. Run Step 2 first.")

    _emit(progress, 75, "Predicting pipeline motion and strains")
    nez = state["NEZ"]
    data = state["data"]
    matched = state["matched"]
    pipe = predict_pipe_motion_and_strain(
        config=config,
        pipe_axis=data["pipe_axis"],
        pipe_axis_segments=data.get("pipe_axis_segments"),
        fault_line=state["notebook_fault_ref"]["fault_line_truth"],
        left_area=state["notebook_fault_ref"]["left_area"],
        fault_polyline_for_projection=np.asarray(state["notebook_fault_ref"]["fl2"]),
        models_by_epoch=state["models_by_epoch"],
        tangential_fit_by_epoch=state.get("tangential_fit_by_epoch", {}),
        normal_fit_by_epoch=state.get("normal_fit_by_epoch", {}),
        fit_overrides_by_epoch=state.get("fault_fit_override_by_epoch", {}),
        pair_count=nez.shape[0],
        intersection_point=state["fit_band"]["intersection"],
    )
    if not bool(getattr(config, "ssi_enabled", False)):
        _emit(progress, 82, "Propagating epistemic cases (b16/b50/b84)")
        epi_cases_by_mode = {
            "Fault-parallel/perpendicular": _run_epistemic_cases(
                config=config,
                state=state,
                pair_count=nez.shape[0],
                include_strain=True,
                prediction_mode="Fault-parallel/perpendicular",
            )
        }
    else:
        # Restore: remove this else and the SSI guard above to re-enable b16/b50/b84.
        _emit(progress, 82, "Skipping epistemic cases (SSI enabled)")
        epi_cases_by_mode = {}
    active_mode = str(getattr(config, "prediction_coordinate_system", "Fault-parallel/perpendicular"))
    epi_cases = epi_cases_by_mode.get(active_mode, {})
    pipe["epistemic_cases"] = epi_cases
    pipe["epistemic_cases_by_mode"] = epi_cases_by_mode

    _emit(progress, 90, "Preparing outputs")
    epi_rows = []
    for epoch_idx, entry in state.get("epistemic_b_by_epoch", {}).items():
        epi_rows.append(
            {
                "epoch_index": int(epoch_idx),
                "b50": float(entry.get("b50", np.nan)),
                "sigma_b": float(entry.get("sigma_b", np.nan)),
                "b16": float(entry.get("b16", np.nan)),
                "b84": float(entry.get("b84", np.nan)),
                "b16_disabled": bool(entry.get("b16_disabled", False)),
                "shear_band_width_m": float(entry.get("shear_band_width_m", np.nan)),
            }
        )
    strain_epi_rows = []
    if all(k in epi_cases for k in ("b16", "b50", "b84")):
        ax16 = np.asarray(epi_cases["b16"]["axial_strain_smoothed"], dtype=float) * 1e6
        ax50 = np.asarray(epi_cases["b50"]["axial_strain_smoothed"], dtype=float) * 1e6
        ax84 = np.asarray(epi_cases["b84"]["axial_strain_smoothed"], dtype=float) * 1e6
        n_epoch = min(ax16.shape[0], ax50.shape[0], ax84.shape[0]) if ax16.ndim == ax50.ndim == ax84.ndim == 2 else 0
        for i in range(n_epoch):
            max_e16 = float(np.nanmax(ax16[i]))
            max_e50 = float(np.nanmax(ax50[i]))
            max_e84 = float(np.nanmax(ax84[i]))
            sigma_epi = _compute_final_epistemic_sigma(max_e16, max_e50, max_e84)
            strain_epi_rows.append(
                {
                    "epoch_index": int(i),
                    "max_eps16_microstrain": max_e16,
                    "max_eps50_microstrain": max_e50,
                    "max_eps84_microstrain": max_e84,
                    "epistemic_sigma_lognormal": sigma_epi,
                }
            )

    tables = {
        "matched_dates": pd.DataFrame(matched["matched_pairs"], columns=["asc_date", "desc_date"]),
        "pipe_axial_latest": pd.DataFrame(
            {
                "x": np.asarray(pipe.get("pipe_axis_nodes", data["pipe_axis"]))[:, 0],
                "y": np.asarray(pipe.get("pipe_axis_nodes", data["pipe_axis"]))[:, 1],
                "s_nodes_offset_m": pipe["s_nodes_offset"],
                "axial_strain_latest": pipe["axial_strain"][-1],
                "axial_strain_smoothed_latest": pipe["axial_strain_smoothed"][-1],
            }
        ),
        "pipe_bending_latest": pd.DataFrame(
            {
                "x": pipe["pipe_axis_sampled"][:, 0],
                "y": pipe["pipe_axis_sampled"][:, 1],
                "bending_strain_latest": pipe["bending_strain"][-1],
            }
        ),
        "epistemic_b_summary": pd.DataFrame(epi_rows),
    }
    if strain_epi_rows:
        tables["strain_epistemic_summary"] = pd.DataFrame(strain_epi_rows)

    arrays = {
        "NEZ": nez,
        "matched_coords": matched["matched_coords"],
        "signed_distance": state["signed_distance"],
        "pipe_motion_xy_mm": pipe["pipe_motion_xy_mm"],
        "pipe_motion_fault_mm": pipe["pipe_motion_fault_mm"],
        "soil_motion_xy_mm": np.asarray(pipe.get("soil_motion_xy_mm", []), dtype=float),
        "soil_motion_fault_mm": np.asarray(pipe.get("soil_motion_fault_mm", []), dtype=float),
        "axial_strain": pipe["axial_strain"],
        "axial_strain_smoothed": pipe["axial_strain_smoothed"],
        "bending_strain": pipe["bending_strain"],
        "day_diffs": matched["day_diffs"],
    }
    for case_key, pred in epi_cases.items():
        arrays[f"pipe_motion_fault_{case_key}_mm"] = np.asarray(pred.get("pipe_motion_fault_mm", []), dtype=float)
        arrays[f"pipe_motion_xy_{case_key}_mm"] = np.asarray(pred.get("pipe_motion_xy_mm", []), dtype=float)
        arrays[f"axial_strain_smoothed_{case_key}"] = np.asarray(pred.get("axial_strain_smoothed", []), dtype=float)
        arrays[f"bending_strain_{case_key}"] = np.asarray(pred.get("bending_strain", []), dtype=float)
    for mode_key, mode_cases in epi_cases_by_mode.items():
        mode_slug = "ne" if mode_key == "N&E" else "fault"
        for case_key, pred in mode_cases.items():
            arrays[f"pipe_motion_xy_{mode_slug}_{case_key}_mm"] = np.asarray(pred.get("pipe_motion_xy_mm", []), dtype=float)
            arrays[f"pipe_motion_fault_{mode_slug}_{case_key}_mm"] = np.asarray(
                pred.get("pipe_motion_fault_mm", []), dtype=float
            )
            arrays[f"axial_strain_smoothed_{mode_slug}_{case_key}"] = np.asarray(
                pred.get("axial_strain_smoothed", []), dtype=float
            )
            arrays[f"bending_strain_{mode_slug}_{case_key}"] = np.asarray(pred.get("bending_strain", []), dtype=float)
    if strain_epi_rows:
        state["epistemic_final_sigma_by_epoch"] = np.asarray(
            [row["epistemic_sigma_lognormal"] for row in strain_epi_rows], dtype=float
        )
    state["pipe_epistemic_cases"] = epi_cases
    state["pipe_epistemic_cases_by_mode"] = epi_cases_by_mode
    state["max_abs_xy_diff_between_modes_mm"] = _max_abs_xy_diff_between_modes(epi_cases_by_mode)
    state.update({"pipe": pipe, "tables": tables, "arrays": arrays})
    _emit(progress, 100, "Step 4 completed")
    return state


def run_step_displacement(
    config: AppConfig,
    state: Dict,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    required = ["models_by_epoch", "NEZ", "data", "matched", "signed_distance"]
    if any(k not in state for k in required):
        raise ValueError("Step 3 requires Step 2 results. Run Step 2 first.")

    _emit(progress, 75, "Predicting pipeline displacement")
    nez = state["NEZ"]
    data = state["data"]
    matched = state["matched"]
    pipe_all = predict_pipe_motion_and_strain(
        config=config,
        pipe_axis=data["pipe_axis"],
        pipe_axis_segments=data.get("pipe_axis_segments"),
        fault_line=state["notebook_fault_ref"]["fault_line_truth"],
        left_area=state["notebook_fault_ref"]["left_area"],
        fault_polyline_for_projection=np.asarray(state["notebook_fault_ref"]["fl2"]),
        models_by_epoch=state["models_by_epoch"],
        tangential_fit_by_epoch=state.get("tangential_fit_by_epoch", {}),
        normal_fit_by_epoch=state.get("normal_fit_by_epoch", {}),
        fit_overrides_by_epoch=state.get("fault_fit_override_by_epoch", {}),
        pair_count=nez.shape[0],
        intersection_point=state["fit_band"]["intersection"],
        compute_strain=False,
    )
    pipe = {
        "distance_pipe": pipe_all["distance_pipe"],
        "pipe_motion_xy_mm": pipe_all["pipe_motion_xy_mm"],
        "pipe_motion_fault_mm": pipe_all["pipe_motion_fault_mm"],
        "soil_motion_xy_mm": pipe_all.get("soil_motion_xy_mm", pipe_all["pipe_motion_xy_mm"]),
        "soil_motion_fault_mm": pipe_all.get("soil_motion_fault_mm", pipe_all["pipe_motion_fault_mm"]),
        "pipe_axis_nodes": pipe_all["pipe_axis_nodes"],
        "pipe_axis_segments": pipe_all.get("pipe_axis_segments", []),
        "pipe_connectivity": pipe_all.get("pipe_connectivity", np.empty((0, 2), dtype=int)),
        "pipe_segment_ids": pipe_all.get("pipe_segment_ids", np.array([], dtype=int)),
        "pipe_axis_sampled": pipe_all["pipe_axis_sampled"],
        "pipe_axis_sampled_segment_ids": pipe_all.get("pipe_axis_sampled_segment_ids", np.array([], dtype=int)),
        "s_nodes_offset": pipe_all["s_nodes_offset"],
        "ssi_enabled": bool(pipe_all.get("ssi_enabled", False)),
        "ssi_zone_by_epoch": np.asarray(pipe_all.get("ssi_zone_by_epoch", []), dtype=bool),
        "ssi_transition_by_epoch": pipe_all.get("ssi_transition_by_epoch", []),
        "prediction_coordinate_system": pipe_all["prediction_coordinate_system"],
    }
    if not bool(getattr(config, "ssi_enabled", False)):
        _emit(progress, 82, "Propagating epistemic cases (b16/b50/b84)")
        epi_cases_by_mode = {
            "Fault-parallel/perpendicular": _run_epistemic_cases(
                config=config,
                state=state,
                pair_count=nez.shape[0],
                include_strain=False,
                prediction_mode="Fault-parallel/perpendicular",
            )
        }
    else:
        # Restore: remove this else and the SSI guard above to re-enable b16/b50/b84.
        _emit(progress, 82, "Skipping epistemic cases (SSI enabled)")
        epi_cases_by_mode = {}
    active_mode = str(getattr(config, "prediction_coordinate_system", "Fault-parallel/perpendicular"))
    epi_cases = epi_cases_by_mode.get(active_mode, {})
    pipe["epistemic_cases"] = epi_cases
    pipe["epistemic_cases_by_mode"] = epi_cases_by_mode

    _emit(progress, 90, "Preparing displacement outputs")
    epi_rows = []
    for epoch_idx, entry in state.get("epistemic_b_by_epoch", {}).items():
        epi_rows.append(
            {
                "epoch_index": int(epoch_idx),
                "b50": float(entry.get("b50", np.nan)),
                "sigma_b": float(entry.get("sigma_b", np.nan)),
                "b16": float(entry.get("b16", np.nan)),
                "b84": float(entry.get("b84", np.nan)),
                "b16_disabled": bool(entry.get("b16_disabled", False)),
                "shear_band_width_m": float(entry.get("shear_band_width_m", np.nan)),
            }
        )
    tables = {
        "matched_dates": pd.DataFrame(matched["matched_pairs"], columns=["asc_date", "desc_date"]),
        "pipe_displacement_latest": pd.DataFrame(
            {
                "x": np.asarray(pipe.get("pipe_axis_nodes", data["pipe_axis"]))[:, 0],
                "y": np.asarray(pipe.get("pipe_axis_nodes", data["pipe_axis"]))[:, 1],
                "s_nodes_offset_m": pipe["s_nodes_offset"],
                "east_mm_latest": pipe["pipe_motion_xy_mm"][-1, :, 0],
                "north_mm_latest": pipe["pipe_motion_xy_mm"][-1, :, 1],
            }
        ),
        "epistemic_b_summary": pd.DataFrame(epi_rows),
    }
    arrays = {
        "NEZ": nez,
        "matched_coords": matched["matched_coords"],
        "signed_distance": state["signed_distance"],
        "pipe_motion_xy_mm": pipe["pipe_motion_xy_mm"],
        "pipe_motion_fault_mm": pipe["pipe_motion_fault_mm"],
        "soil_motion_xy_mm": np.asarray(pipe.get("soil_motion_xy_mm", []), dtype=float),
        "soil_motion_fault_mm": np.asarray(pipe.get("soil_motion_fault_mm", []), dtype=float),
        "day_diffs": matched["day_diffs"],
    }
    for case_key, pred in epi_cases.items():
        arrays[f"pipe_motion_fault_{case_key}_mm"] = np.asarray(pred.get("pipe_motion_fault_mm", []), dtype=float)
        arrays[f"pipe_motion_xy_{case_key}_mm"] = np.asarray(pred.get("pipe_motion_xy_mm", []), dtype=float)
    for mode_key, mode_cases in epi_cases_by_mode.items():
        mode_slug = "ne" if mode_key == "N&E" else "fault"
        for case_key, pred in mode_cases.items():
            arrays[f"pipe_motion_xy_{mode_slug}_{case_key}_mm"] = np.asarray(pred.get("pipe_motion_xy_mm", []), dtype=float)
            arrays[f"pipe_motion_fault_{mode_slug}_{case_key}_mm"] = np.asarray(
                pred.get("pipe_motion_fault_mm", []), dtype=float
            )
    state["pipe_epistemic_cases"] = epi_cases
    state["pipe_epistemic_cases_by_mode"] = epi_cases_by_mode
    state["max_abs_xy_diff_between_modes_mm"] = _max_abs_xy_diff_between_modes(epi_cases_by_mode)
    state.update({"pipe": pipe, "tables": tables, "arrays": arrays})
    _emit(progress, 100, "Step 3 completed")
    return state


def run_pipeline(config: AppConfig, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    def emit(pct, msg):
        if progress:
            progress(pct, msg)

    state = run_step_match_3d(config, state=None, progress=lambda p, m: emit(min(p, 45), m))
    state = run_step_fit(config, state, progress=lambda p, m: emit(45 + int(p * 0.3), m))
    state = run_step_strain(config, state, progress=lambda p, m: emit(75 + int((p - 75) * 1.0), m))
    emit(100, "Completed")
    return state
