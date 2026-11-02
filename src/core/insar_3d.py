from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.config import AppConfig


def _date_columns(df, prefix: str = "D") -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No date columns with prefix '{prefix}' found.")
    return cols


def _parse_date_col(col: str) -> datetime:
    return datetime.strptime(col[1:], "%Y%m%d")


def match_insar(config: AppConfig, asc_data, desc_data) -> Dict:
    asc_dates = _date_columns(asc_data, config.date_column_prefix)
    desc_dates = _date_columns(desc_data, config.date_column_prefix)

    matched_pairs: List[Tuple[str, str]] = []
    for asc_col in asc_dates:
        asc_dt = _parse_date_col(asc_col)
        closest = min(desc_dates, key=lambda x: abs(_parse_date_col(x) - asc_dt))
        if abs((_parse_date_col(closest) - asc_dt).days) <= config.date_match_max_days:
            matched_pairs.append((asc_col, closest))

    if not matched_pairs:
        raise ValueError("No asc/desc date pairs satisfy date_match_max_days.")

    asc_coords = np.array([(geom.x, geom.y) for geom in asc_data.geometry])
    desc_coords = np.array([(geom.x, geom.y) for geom in desc_data.geometry])

    tree = cKDTree(desc_coords)
    distances, indices = tree.query(asc_coords, distance_upper_bound=config.spatial_match_max_m)
    valid = distances < config.spatial_match_max_m
    if valid.sum() == 0:
        raise ValueError("No asc/desc point pairs satisfy spatial_match_max_m.")

    matched_asc_data = asc_data.loc[valid].reset_index(drop=True)
    matched_desc_data = desc_data.iloc[indices[valid]].reset_index(drop=True)
    matched_coords = asc_coords[valid]

    base_date = _parse_date_col(matched_pairs[0][0])
    day_diffs = np.array([(_parse_date_col(d1) - base_date).days for d1, _ in matched_pairs], dtype=int)

    return {
        "matched_pairs": matched_pairs,
        "matched_asc_data": matched_asc_data,
        "matched_desc_data": matched_desc_data,
        "matched_coords": matched_coords,
        "day_diffs": day_diffs,
    }


def _los_unit_vector(heading_deg: float, inc_deg: float) -> np.ndarray:
    return np.array(
        [
            -np.sin(np.radians(inc_deg)) * np.sin(np.radians(heading_deg)),
            np.sin(np.radians(inc_deg)) * np.cos(np.radians(heading_deg)),
            -np.cos(np.radians(inc_deg)),
        ],
        dtype=float,
    )


def constraint_vector(config: AppConfig) -> np.ndarray:
    if config.constraint_mode == "aspect_direction":
        asp = config.constraint_aspect_deg
        return np.array([-np.sin(np.radians(asp)), np.cos(np.radians(asp)), 0.0], dtype=float)
    if config.constraint_mode == "north_zero":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if config.constraint_mode == "east_zero":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if config.constraint_mode == "vertical_zero":
        return np.array([0.0, 0.0, 1.0], dtype=float)
    raise ValueError(f"Unknown constraint_mode: {config.constraint_mode}")


def solve_3d_displacement(config: AppConfig, matched: Dict) -> Dict:
    l_asc = _los_unit_vector(config.asc_heading_deg, config.asc_inc_deg)
    l_desc = _los_unit_vector(config.desc_heading_deg, config.desc_inc_deg)
    l_cstrn = constraint_vector(config)
    l = np.vstack([l_asc, l_desc, l_cstrn])

    cond = np.linalg.cond(l)
    pinv = np.linalg.pinv(l)

    asc_df = matched["matched_asc_data"]
    desc_df = matched["matched_desc_data"]
    pairs = matched["matched_pairs"]

    n_epoch = len(pairs)
    n_pt = len(asc_df)
    nez = np.zeros((n_epoch, n_pt, 3), dtype=float)

    for j, (asc_col, desc_col) in enumerate(pairs):
        # Notebook logic: L = [-asc, -desc, 0], then NEZ = lstsq(l, L)
        L = np.vstack(
            [
                -asc_df[asc_col].to_numpy(dtype=float),
                -desc_df[desc_col].to_numpy(dtype=float),
                np.zeros(n_pt, dtype=float),
            ]
        )
        nez[j] = (pinv @ L).T

    return {
        "NEZ": nez,
        "diagnostics": {
            "los_matrix": l,
            "condition_number": float(cond),
            "ill_conditioned": bool(cond > 1e3),
        },
    }
