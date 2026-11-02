from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


def software_root() -> Path:
    # src/config.py -> project root (same folder as main.py)
    return Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    # A. Input files
    asc_shp_path: str = "../../Data/InSAR/Claremont_Ave_Ascending/Claremont_A76.shp"
    desc_shp_path: str = "../../Data/InSAR/Claremont_Ave_Dscending/Claremont_D38.shp"
    combined_shp_path: str = "../../Data/InSAR/Claremont_InSAR-EW/Claremont.shp"
    fault_line_shp_path: str = "AP_fault//all_hayward.shp"
    ml_fault_polyline_npy: str = "bound_rep.npy"
    pipe_axis_path: str = "pipe_axis_smooth_v7_3k.npy"

    # Advanced optional files
    mma_npy_path: str = "mma.npy"
    fiber_csv_path: str = "claremont_yaobin//ClaremontNorth36_Z_tc.csv"

    # Output
    output_dir: str = "."

    # B. Matching
    date_match_max_days: int = 4
    spatial_match_max_m: float = 5.0
    processing_crs: str = "EPSG:3857"
    pipeline_input_crs: str = "EPSG:3857"

    # C. LOS + constraint
    asc_heading_deg: float = 349.623233
    asc_inc_deg: float = 33.153514
    desc_heading_deg: float = 191.303877
    desc_inc_deg: float = 21.43727
    constraint_mode: str = "aspect_direction"  # aspect_direction | north_zero | east_zero | vertical_zero
    constraint_aspect_deg: float = 148.63
    constraint_aspect_source: str = "manual"  # manual | from_selected_fault_linear_fit
    constraint_b16_floor: float = 0.01
    flip_signed_distance: bool = True

    # D. Fault geometry
    fault_trace_type: str = "selected_geometry"  # selected_geometry | linear_fit | quadratic_fit
    fault_picker_bbox_minx: float = -13622000.0
    fault_picker_bbox_maxx: float = -13617000.0
    fault_picker_bbox_miny: float = 4573000.0
    fault_picker_bbox_maxy: float = 4581000.0
    fault_fit_sample_count: int = 900
    fault_picker_click_max_dist_m: float = 200.0
    pipe_clip_half_length_m: float = 300.0
    fault_simplify_tolerance: float = 5.0
    side_polygon_margin_m: float = 500.0
    transect_half_length_m: float = 1600.0
    fit_band_offset_m: float = 120.0
    offset_h_m: float = 100.0
    offset_v_m: float = 73.71884299209695

    # E. Map view
    map_xlim_min: float = -13608455.593597125
    map_xlim_max: float = -13607499.116105666
    map_ylim_min: float = 4559228.09352576
    map_ylim_max: float = 4560088.3680825755

    # F. Model function
    model_type: str = "sigmoid"  # sigmoid | piecewise_linear
    prediction_coordinate_system: str = "Fault-parallel/perpendicular"  # N&E | Fault-parallel/perpendicular
    sigmoid_a: float = 39.91972585
    sigmoid_b: float = 0.61268
    sigmoid_c: float = 0.0
    sigmoid_d: float = -6.07378208

    # G. Bending
    bending_sample_step: int = 2
    axial_smoothing_window: int = 0
    mesh_size_m: float = 8.0

    # G2. SSI
    ssi_enabled: bool = False
    ssi_D_out_m: float = 0.9144
    ssi_t_wall_m: float = 0.08313
    ssi_E_pipe_kN_m2: float = 1000000.0
    ssi_bc_type: str = "pinned-pinned"  # fixed-fixed | pinned-pinned | fixed-free | pinned-free
    ssi_E_soil_outer_kN_m2: float = 10000.0
    ssi_E_soil_center_kN_m2: float = 2000.0
    ssi_k2_ratio_outer: float = 0.325
    ssi_k2_ratio_center: float = 0.325
    ssi_mu_friction: float = 0.678
    ssi_delta_y_m: float = 0.001
    ssi_delta_max_m: float = 0.023
    ssi_k_axial_kN_m: float = 10000.0
    ssi_depth_burial_m: float = 1.5
    ssi_gamma_soil_kN_m3: float = 18.0
    ssi_k_gap_penalty_kN_m: float = 5.0e5
    ssi_n_increments: int = 20
    ssi_max_iter: int = 50
    ssi_tol: float = 1e-6

    # H. Optional behavior controls
    auto_use_ml_fault_polyline: bool = True
    date_column_prefix: str = "D"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        patched = dict(data or {})
        # Backward compatibility for old saved states/configs.
        if "processing_crs" not in patched and "map_epsg" in patched:
            old = patched.get("map_epsg")
            if isinstance(old, str) and old.upper().startswith("EPSG:"):
                patched["processing_crs"] = old.upper()
            else:
                patched["processing_crs"] = f"EPSG:{int(old)}"
        valid = cls().__dict__.keys()
        filtered = {k: v for k, v in patched.items() if k in valid}
        if "processing_crs" in filtered:
            crs = str(filtered["processing_crs"]).strip()
            filtered["processing_crs"] = crs.upper() if crs else "EPSG:3857"
        if "pipeline_input_crs" in filtered:
            pcrs = str(filtered["pipeline_input_crs"]).strip()
            filtered["pipeline_input_crs"] = pcrs.upper() if pcrs else "EPSG:3857"
        if "constraint_aspect_source" in filtered:
            src = str(filtered["constraint_aspect_source"]).strip().lower()
            filtered["constraint_aspect_source"] = src if src else "manual"
        if "ssi_bc_type" in filtered:
            bc = str(filtered["ssi_bc_type"]).strip().lower()
            valid_bc = {"fixed-fixed", "pinned-pinned", "fixed-free", "pinned-free"}
            filtered["ssi_bc_type"] = bc if bc in valid_bc else "pinned-pinned"
        return cls(**filtered)

    def resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path).expanduser()
        if p.is_absolute():
            return p
        return (software_root() / p).resolve()
