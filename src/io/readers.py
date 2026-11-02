from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from src.config import AppConfig


def _resolve_processing_epsg(config: AppConfig) -> int:
    raw = str(getattr(config, "processing_crs", "EPSG:3857")).strip().upper()
    if raw.startswith("EPSG:"):
        return int(raw.split(":", 1)[1])
    return int(raw)


def _resolve_pipeline_input_epsg(config: AppConfig) -> int:
    raw = str(getattr(config, "pipeline_input_crs", "EPSG:3857")).strip().upper()
    if raw.startswith("EPSG:"):
        return int(raw.split(":", 1)[1])
    return int(raw)


def _reproject_xy_array(points_xy: np.ndarray, src_epsg: int, dst_epsg: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return pts
    if int(src_epsg) == int(dst_epsg):
        return pts
    gs = gpd.GeoSeries(gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs=f"EPSG:{int(src_epsg)}")
    gs2 = gs.to_crs(epsg=int(dst_epsg))
    return np.column_stack([gs2.x.to_numpy(dtype=float), gs2.y.to_numpy(dtype=float)])


def _ensure_gdf_epsg(gdf: gpd.GeoDataFrame, epsg: int, layer_name: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError(f"{layer_name} shapefile has no CRS metadata; cannot transform to EPSG:{epsg}.")
    return gdf.to_crs(epsg=epsg)


def _pick_longest_line(geom) -> Optional[LineString]:
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
        return max(parts, key=lambda g: g.length) if parts else None
    return None


def _line_from_gdf(gdf: gpd.GeoDataFrame) -> LineString:
    # Pass a plain list of LineStrings: shapely>=2.1 linemerge() raises
    # "truth value of an array is ambiguous" when given a GeometryArray.
    merged = linemerge(_flatten_lines(gdf.geometry.values))
    line = _pick_longest_line(merged)
    if line is None:
        lines = [g for g in gdf.geometry.values if isinstance(g, (LineString, MultiLineString))]
        if not lines:
            raise ValueError("Fault shapefile contains no LineString geometry.")
        line = _pick_longest_line(max(lines, key=lambda g: g.length))
    if line is None:
        raise ValueError("Cannot extract fault polyline from shapefile.")
    return line


def _load_pipe_axis(
    path: Path,
    target_epsg: Optional[int] = None,
    source_epsg_for_nonshp: Optional[int] = None,
) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
        pts = np.asarray(arr)[:, :2]
        if target_epsg is not None and source_epsg_for_nonshp is not None:
            pts = _reproject_xy_array(pts, source_epsg_for_nonshp, target_epsg)
        return pts
    if ext in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        pts = np.asarray(arr)[:, :2]
        if target_epsg is not None and source_epsg_for_nonshp is not None:
            pts = _reproject_xy_array(pts, source_epsg_for_nonshp, target_epsg)
        return pts
    if ext == ".shp":
        pipe_gdf = gpd.read_file(path)
        if target_epsg is not None:
            pipe_gdf = _ensure_gdf_epsg(pipe_gdf, int(target_epsg), "Pipe")
        if pipe_gdf.empty:
            raise ValueError("Pipe shapefile is empty.")
        geom = pipe_gdf.geometry.iloc[0]
        if hasattr(geom, "coords"):
            return np.asarray(geom.coords)[:, :2]
        raise ValueError("Pipe shapefile first geometry has no coordinate sequence.")
    raise ValueError(f"Unsupported pipe axis format: {path}")


def _flatten_lines(geoms) -> list:
    lines = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            lines.append(geom)
        elif isinstance(geom, MultiLineString):
            for ln in geom.geoms:
                if ln is not None and not ln.is_empty:
                    lines.append(ln)
    return lines


def _load_pipe_lines(
    path: Path,
    target_epsg: Optional[int] = None,
    source_epsg_for_nonshp: Optional[int] = None,
) -> list:
    ext = path.suffix.lower()
    if ext == ".shp":
        pipe_gdf = gpd.read_file(path)
        if target_epsg is not None:
            pipe_gdf = _ensure_gdf_epsg(pipe_gdf, int(target_epsg), "Pipe")
        if pipe_gdf.empty:
            raise ValueError("Pipe shapefile is empty.")
        lines = _flatten_lines(pipe_gdf.geometry.values)
        if not lines:
            raise ValueError("Pipe shapefile contains no valid line geometry.")
        return lines
    axis = _load_pipe_axis(
        path,
        target_epsg=target_epsg,
        source_epsg_for_nonshp=source_epsg_for_nonshp,
    )
    if axis.ndim != 2 or axis.shape[0] < 2:
        raise ValueError("Pipe axis has insufficient nodes.")
    return [LineString(np.asarray(axis, dtype=float)[:, :2])]


def load_inputs(config: AppConfig) -> Dict:
    asc_path = config.resolve_path(config.asc_shp_path)
    desc_path = config.resolve_path(config.desc_shp_path)
    combined_path = config.resolve_path(config.combined_shp_path)
    fault_shp_path = config.resolve_path(config.fault_line_shp_path)
    fault_npy_path = config.resolve_path(config.ml_fault_polyline_npy)
    pipe_path = config.resolve_path(config.pipe_axis_path)

    missing = [p for p in [asc_path, desc_path, fault_shp_path, pipe_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(p) for p in missing))

    target_epsg = _resolve_processing_epsg(config)
    pipe_src_epsg = _resolve_pipeline_input_epsg(config)
    asc_data = _ensure_gdf_epsg(gpd.read_file(asc_path), target_epsg, "Ascending InSAR")
    desc_data = _ensure_gdf_epsg(gpd.read_file(desc_path), target_epsg, "Descending InSAR")
    combined_data = (
        _ensure_gdf_epsg(gpd.read_file(combined_path), target_epsg, "Combined InSAR")
        if combined_path.exists()
        else None
    )
    fault_data = _ensure_gdf_epsg(gpd.read_file(fault_shp_path), target_epsg, "Fault")

    fault_polyline = None
    if config.auto_use_ml_fault_polyline and fault_npy_path.exists():
        fault_arr = np.load(fault_npy_path)
        if fault_arr.ndim != 2 or fault_arr.shape[1] < 2:
            raise ValueError(f"Fault polyline npy invalid shape: {fault_arr.shape}")
        fault_polyline = LineString(np.asarray(fault_arr)[:, :2])
    if fault_polyline is None:
        fault_polyline = _line_from_gdf(fault_data)

    pipe_lines_all = _load_pipe_lines(
        pipe_path,
        target_epsg=target_epsg,
        source_epsg_for_nonshp=pipe_src_epsg,
    )
    merged_pipe = linemerge(pipe_lines_all)
    pipe_main = _pick_longest_line(merged_pipe)
    if pipe_main is None:
        pipe_main = max(pipe_lines_all, key=lambda g: g.length)
    pipe_axis = np.asarray(pipe_main.coords, dtype=float)[:, :2]

    def _read_optional_npy(raw: str):
        p = config.resolve_path(raw)
        return np.load(p) if p.exists() else None

    def _read_optional_csv(raw: str):
        p = config.resolve_path(raw)
        if not p.exists():
            return None
        return pd.read_csv(p)

    return {
        "asc_data": asc_data,
        "desc_data": desc_data,
        "combined_data": combined_data,
        "fault_data": fault_data,
        "fault_polyline": fault_polyline,
        "pipe_axis": pipe_axis,
        "pipe_lines_all": pipe_lines_all,
        "mma": _read_optional_npy(config.mma_npy_path),
        "fiber": _read_optional_csv(config.fiber_csv_path),
    }
