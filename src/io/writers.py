from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig


def make_run_dir(output_dir: str) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(config: AppConfig, run_dir: Path) -> Path:
    out = run_dir / "config.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
    return out


def save_numpy_arrays(arrays: Dict[str, np.ndarray], run_dir: Path) -> Iterable[Path]:
    outputs = []
    for key, value in arrays.items():
        p = run_dir / f"{key}.npy"
        np.save(p, value)
        outputs.append(p)
    return outputs


def save_csv_tables(tables: Dict[str, pd.DataFrame], run_dir: Path) -> Iterable[Path]:
    outputs = []
    for key, df in tables.items():
        p = run_dir / f"{key}.csv"
        df.to_csv(p, index=False)
        outputs.append(p)
    return outputs


def save_figures(figures: Dict[str, object], run_dir: Path) -> Iterable[Path]:
    outputs = []
    for key, fig in figures.items():
        p = run_dir / f"{key}.png"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        outputs.append(p)
    return outputs


def export_pipeline_results(
    config: AppConfig,
    arrays: Dict[str, np.ndarray],
    tables: Dict[str, pd.DataFrame],
    figures: Optional[Dict[str, object]] = None,
) -> Path:
    run_dir = make_run_dir(config.output_dir)
    save_config(config, run_dir)
    list(save_numpy_arrays(arrays, run_dir))
    list(save_csv_tables(tables, run_dir))
    if figures:
        list(save_figures(figures, run_dir))
    return run_dir
