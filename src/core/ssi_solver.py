from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SSIParams:
    D_out_m: float
    t_wall_m: float
    E_pipe_kN_m2: float
    bc_type: str
    E_soil_outer_kN_m2: float
    E_soil_center_kN_m2: float
    k2_ratio_outer: float
    k2_ratio_center: float
    mu_friction: float
    delta_y_m: float
    delta_max_m: float
    k_axial_kN_m: float
    depth_burial_m: float
    gamma_soil_kN_m3: float
    k_gap_penalty_kN_m: float
    n_increments: int
    max_iter: int
    tol: float


def _compute_lateral_params(
    E_soil: float,
    D_out: float,
    spacing: float,
    delta_y: float,
    delta_max: float,
    k2_ratio: float,
) -> Dict[str, float]:
    k_per_length = max(float(E_soil), 1e-6) * max(float(D_out), 1e-6)
    k0 = k_per_length * max(float(spacing), 1e-6)
    dy = max(float(delta_y), 1e-7)
    dm = max(float(delta_max), dy + 1e-7)
    k2 = max(float(k2_ratio), 0.0) * k0
    f_y = k0 * dy
    f_max = f_y + k2 * max(dm - dy, 0.0)
    return {"k0": k0, "k2": k2, "delta_y": dy, "delta_max": dm, "F_y": f_y, "F_max": f_max}


def _lateral_force_tangent(delta: float, prm: Dict[str, float]) -> tuple[float, float]:
    k0 = prm["k0"]
    k2 = prm["k2"]
    dy = prm["delta_y"]
    dm = prm["delta_max"]
    f_y = prm["F_y"]
    fm = prm["F_max"]
    ad = abs(float(delta))
    sgn = 0.0 if ad <= 1e-15 else np.sign(delta)
    if ad <= dy:
        return k0 * delta, k0
    if ad <= dm:
        return sgn * (f_y + k2 * (ad - dy)), k2
    return sgn * fm, 0.0


def _axial_force_tangent(delta: float, k_axial: float, friction_cap: float) -> tuple[float, float]:
    # Modified Newton: clip the axial force at the friction cap, but keep the
    # iteration tangent equal to the elastic spring stiffness. This is not the
    # Abaqus default consistent tangent.
    k_el = max(float(k_axial), 1e-12)
    f_el = k_el * float(delta)
    cap = abs(float(friction_cap))
    if abs(f_el) <= cap:
        return f_el, k_el
    sgn = float(np.sign(delta)) if abs(float(delta)) > 1e-15 else 0.0
    return sgn * cap, k_el


def _assemble_two_node_spring_chain(n: int, spacing: np.ndarray, ke_of_h) -> np.ndarray:
    k_pipe = np.zeros((max(int(n), 0), max(int(n), 0)), dtype=float)
    if n < 2:
        return k_pipe
    ds = np.asarray(spacing, dtype=float).reshape(-1)
    n_elem = min(n - 1, int(ds.size))
    for e in range(n_elem):
        h = max(float(ds[e]), 1e-6)
        ke = float(ke_of_h(h))
        if not np.isfinite(ke) or ke <= 0.0:
            continue
        k_pipe[e, e] += ke
        k_pipe[e, e + 1] -= ke
        k_pipe[e + 1, e] -= ke
        k_pipe[e + 1, e + 1] += ke
    return k_pipe


def _make_axial_pipe_operator(n: int, spacing: np.ndarray, EA: float) -> np.ndarray:
    ea = max(float(EA), 0.0)
    return _assemble_two_node_spring_chain(n, spacing, lambda h: ea / h)


def _euler_element_ke(h: float, ei: float) -> np.ndarray:
    # Standard Euler-Bernoulli bending stiffness for [v1, theta1, v2, theta2].
    h = max(float(h), 1e-12)
    ei = max(float(ei), 0.0)
    c = ei / (h ** 3)
    return c * np.array(
        [
            [12.0, 6.0 * h, -12.0, 6.0 * h],
            [6.0 * h, 4.0 * h * h, -6.0 * h, 2.0 * h * h],
            [-12.0, -6.0 * h, 12.0, -6.0 * h],
            [6.0 * h, 2.0 * h * h, -6.0 * h, 4.0 * h * h],
        ],
        dtype=float,
    )


def _assemble_euler_beam_bending(n: int, spacing: np.ndarray, EI: float) -> np.ndarray:
    # DOF order: [v0, theta0, v1, theta1, ...].
    n = max(int(n), 0)
    k_beam = np.zeros((2 * n, 2 * n), dtype=float)
    if n < 2:
        return k_beam
    ds = np.asarray(spacing, dtype=float).reshape(-1)
    ei = max(float(EI), 0.0)
    n_elem = min(n - 1, int(ds.size))
    for e in range(n_elem):
        ke = _euler_element_ke(ds[e], ei)
        i0 = 2 * e
        k_beam[i0 : i0 + 4, i0 : i0 + 4] += ke
    return k_beam


def _recover_euler_end_moments(
    v: np.ndarray,
    theta: np.ndarray,
    spacing: np.ndarray,
    EI: float,
) -> tuple[float, float]:
    # Element-end moments at the first and last beam nodes (kN·m).
    v = np.asarray(v, dtype=float).reshape(-1)
    th = np.asarray(theta, dtype=float).reshape(-1)
    n = int(min(v.size, th.size))
    ds = np.asarray(spacing, dtype=float).reshape(-1)
    if n < 2 or ds.size < 1:
        return 0.0, 0.0
    u0 = np.array([v[0], th[0], v[1], th[1]], dtype=float)
    f0 = _euler_element_ke(ds[0], EI) @ u0
    el = n - 2
    h_last = ds[min(el, int(ds.size) - 1)]
    u1 = np.array([v[n - 2], th[n - 2], v[n - 1], th[n - 1]], dtype=float)
    f1 = _euler_element_ke(h_last, EI) @ u1
    return float(f0[1]), float(f1[3])


def _apply_dirichlet(k_mat: np.ndarray, rhs: np.ndarray, idx: int, value: float):
    col = k_mat[:, idx].copy()
    rhs[:] = rhs - col * float(value)
    k_mat[idx, :] = 0.0
    k_mat[:, idx] = 0.0
    k_mat[idx, idx] = 1.0
    rhs[idx] = float(value)


def _apply_bc(k_mat: np.ndarray, rhs: np.ndarray, soil_u: np.ndarray, bc_type: str):
    if k_mat.size == 0:
        return
    kind = str(bc_type).strip().lower()
    fix_left = kind in {"fixed-fixed", "fixed-free", "pinned-pinned", "pinned-free"}
    fix_right = kind in {"fixed-fixed", "pinned-pinned"}
    n = k_mat.shape[0]
    if fix_left and n > 0:
        _apply_dirichlet(k_mat, rhs, 0, float(soil_u[0]))
    if fix_right and n > 1:
        _apply_dirichlet(k_mat, rhs, n - 1, float(soil_u[-1]))


def _apply_beam_bc(k_mat: np.ndarray, rhs: np.ndarray, soil_v: np.ndarray, bc_type: str):
    # Pinned: constrain v only (M=0 is a natural BC). Fixed: constrain v and theta=0.
    if k_mat.size == 0:
        return
    kind = str(bc_type).strip().lower()
    n_node = int(k_mat.shape[0] // 2)
    if n_node <= 0:
        return
    soil_v = np.asarray(soil_v, dtype=float).reshape(-1)
    pin_left_v = kind in {"fixed-fixed", "fixed-free", "pinned-pinned", "pinned-free"}
    fix_left_th = kind in {"fixed-fixed", "fixed-free"}
    pin_right_v = kind in {"fixed-fixed", "pinned-pinned"}
    fix_right_th = kind in {"fixed-fixed"}
    if pin_left_v:
        _apply_dirichlet(k_mat, rhs, 0, float(soil_v[0]) if soil_v.size else 0.0)
    if fix_left_th:
        _apply_dirichlet(k_mat, rhs, 1, 0.0)
    if pin_right_v and n_node > 1:
        _apply_dirichlet(k_mat, rhs, 2 * (n_node - 1), float(soil_v[-1]) if soil_v.size else 0.0)
    if fix_right_th and n_node > 1:
        _apply_dirichlet(k_mat, rhs, 2 * (n_node - 1) + 1, 0.0)


def solve_pipe_ssi_fault_components(
    soil_fault_mm: np.ndarray,
    s_nodes_offset_m: np.ndarray,
    in_center_zone: np.ndarray,
    params: SSIParams,
    tangent_xy: Optional[np.ndarray] = None,
    normal_xy: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Solve SSI response in fault coordinates:
    - component 0: tangential (axial spring + friction cap, modified Newton)
    - component 1: normal (Euler-Bernoulli v+theta + nonlinear lateral springs on v)
    """
    soil_fault_mm = np.asarray(soil_fault_mm, dtype=float)
    s = np.asarray(s_nodes_offset_m, dtype=float).reshape(-1)
    zone = np.asarray(in_center_zone, dtype=bool).reshape(-1)
    if soil_fault_mm.ndim != 2 or soil_fault_mm.shape[1] != 2:
        raise ValueError("soil_fault_mm must have shape (N,2).")
    n = int(min(soil_fault_mm.shape[0], s.size, zone.size))
    if n <= 0:
        raise ValueError("SSI input has no valid nodes.")
    soil_fault_mm = soil_fault_mm[:n]
    s = s[:n]
    zone = zone[:n]
    order = np.argsort(s)
    inv = np.argsort(order)
    soil_sorted = soil_fault_mm[order]
    s_sorted = s[order]
    zone_sorted = zone[order]

    ds_seg = np.diff(s_sorted)
    ds_seg = np.where(np.isfinite(ds_seg) & (ds_seg > 1e-6), ds_seg, np.nan)
    if ds_seg.size == 0:
        spacing_mean = 1.0
    else:
        spacing_mean = float(np.nanmean(ds_seg))
        if not np.isfinite(spacing_mean) or spacing_mean <= 1e-6:
            spacing_mean = 1.0
        ds_seg = np.where(np.isfinite(ds_seg), ds_seg, spacing_mean)
    ds_node = np.full(n, max(spacing_mean, 1e-6), dtype=float)
    if n >= 2 and ds_seg.size >= 1:
        ds_node[0] = max(float(ds_seg[0]), 1e-6)
        ds_node[-1] = max(float(ds_seg[-1]), 1e-6)
        if n > 2:
            ds_mid = 0.5 * (ds_seg[:-1] + ds_seg[1:])
            ds_node[1:-1] = np.maximum(ds_mid.astype(float), 1e-6)

    d_out = max(float(params.D_out_m), 1e-6)
    d_in = max(d_out - 2.0 * max(float(params.t_wall_m), 0.0), 1e-6)
    e_pipe = max(float(params.E_pipe_kN_m2), 1e-6)
    area = np.pi / 4.0 * (d_out**2 - d_in**2)
    inertia = np.pi / 64.0 * (d_out**4 - d_in**4)
    ea = max(e_pipe * max(float(area), 1e-12), 1e-6)
    ei = max(e_pipe * max(float(inertia), 1e-12), 1e-6)
    k_pipe_t = _make_axial_pipe_operator(n, ds_seg, ea)
    k_pipe_n = _assemble_euler_beam_bending(n, ds_seg, ei)
    h_mean = max(float(spacing_mean), 1e-6)
    k_pipe_scale_t = max(ea / h_mean, 1e-9)
    k_floor_t = max(1e-9, 1e-8 * k_pipe_scale_t)

    lat_params = []
    for i in range(n):
        e_soil = params.E_soil_center_kN_m2 if zone_sorted[i] else params.E_soil_outer_kN_m2
        k2_ratio = params.k2_ratio_center if zone_sorted[i] else params.k2_ratio_outer
        lat_params.append(
            _compute_lateral_params(
                E_soil=e_soil,
                D_out=d_out,
                spacing=ds_node[i],
                delta_y=params.delta_y_m,
                delta_max=params.delta_max_m,
                k2_ratio=k2_ratio,
            )
        )

    n_over = (
        max(float(params.gamma_soil_kN_m3), 0.0)
        * max(float(params.depth_burial_m), 0.0)
        * (np.pi * d_out)
        * ds_node
    )
    friction_cap = max(float(params.mu_friction), 0.0) * n_over
    # Keep UI k_axial as direct spring stiffness per node to preserve
    # historical calibration behavior and avoid mesh-amplified hardening.
    k_axial_node = np.full(n, max(float(params.k_axial_kN_m), 1e-9), dtype=float)
    k_gap = max(float(params.k_gap_penalty_kN_m), 0.0)
    # Tiny floor only; axial iterations already keep elastic k_axial in the matrix.
    k_floor_t = max(float(k_floor_t), 1e-8 * float(np.max(k_axial_node)))

    # Start incremental loading from zero displacement and use previous converged
    # increment as warm-start to avoid early overshoot into nonphysical branches.
    u_sorted = np.zeros_like(soil_sorted, dtype=float)
    theta_sorted = np.zeros(n, dtype=float)
    n_inc = max(int(params.n_increments), 1)
    n_iter = max(int(params.max_iter), 1)
    tol = max(float(params.tol), 1e-10)
    axial_iter_counts = np.zeros(n_inc, dtype=int)
    axial_converged = np.zeros(n_inc, dtype=bool)

    for inc in range(1, n_inc + 1):
        alpha = float(inc) / float(n_inc)
        soil_now = soil_sorted * 1e-3 * alpha
        ref_t = max(float(np.linalg.norm(soil_now[:, 0])), 1.0)

        # Axial Newton: u_t only; independent of v, theta, and lateral soil.
        ax_ok = False
        ax_iters = 0
        for _ in range(n_iter):
            ax_iters += 1
            k_diag_t = np.zeros(n, dtype=float)
            c_t = np.zeros(n, dtype=float)
            for i in range(n):
                dt = u_sorted[i, 0] - soil_now[i, 0]
                f_t, kt = _axial_force_tangent(dt, k_axial_node[i], friction_cap[i])
                c_t[i] = f_t - kt * dt
                k_diag_t[i] = kt
            kt_mat = k_pipe_t + np.diag(k_diag_t + k_floor_t)
            rhs_t = k_diag_t * soil_now[:, 0] - c_t
            _apply_bc(kt_mat, rhs_t, soil_now[:, 0], params.bc_type)
            try:
                u_t_new = np.linalg.solve(kt_mat, rhs_t)
            except np.linalg.LinAlgError:
                u_t_new = np.linalg.pinv(kt_mat) @ rhs_t
            du_t = float(np.linalg.norm(u_t_new - u_sorted[:, 0]))
            u_sorted[:, 0] = u_t_new
            if du_t / ref_t <= tol:
                ax_ok = True
                break
        axial_iter_counts[inc - 1] = ax_iters
        axial_converged[inc - 1] = ax_ok

        # Lateral Euler-Bernoulli Newton: v, theta only.
        for _ in range(n_iter):
            k_diag_n = np.zeros(n, dtype=float)
            c_n = np.zeros(n, dtype=float)
            for i in range(n):
                dn = u_sorted[i, 1] - soil_now[i, 1]
                f_n, kn = _lateral_force_tangent(dn, lat_params[i])
                if dn < 0.0:
                    f_n = f_n + k_gap * dn
                    kn = kn + k_gap
                c_n[i] = f_n - kn * dn
                k_diag_n[i] = kn
            kn_mat = k_pipe_n.copy()
            rhs_n = np.zeros(2 * n, dtype=float)
            for i in range(n):
                kn_mat[2 * i, 2 * i] += k_diag_n[i]
                rhs_n[2 * i] = k_diag_n[i] * soil_now[i, 1] - c_n[i]
            _apply_beam_bc(kn_mat, rhs_n, soil_now[:, 1], params.bc_type)
            try:
                u_beam = np.linalg.solve(kn_mat, rhs_n)
            except np.linalg.LinAlgError:
                # Regularize only translational DOFs; keep it negligible vs physical K.
                scale = float(np.max(np.abs(np.diag(k_pipe_n)))) if k_pipe_n.size else 1.0
                scale = max(scale, 1.0)
                for i in range(n):
                    kn_mat[2 * i, 2 * i] += 1e-12 * scale
                try:
                    u_beam = np.linalg.solve(kn_mat, rhs_n)
                except np.linalg.LinAlgError:
                    u_beam = np.linalg.pinv(kn_mat) @ rhs_n
            u_n_new = u_beam[0::2]
            th_new = u_beam[1::2]
            err_n = max(
                float(np.max(np.abs(u_n_new - u_sorted[:, 1]))) if n else 0.0,
                float(h_mean * np.max(np.abs(th_new - theta_sorted))) if n else 0.0,
            )
            u_sorted[:, 1] = u_n_new
            theta_sorted[:] = th_new
            if err_n <= tol:
                break

    pipe_sorted_mm = u_sorted * 1e3
    out = np.zeros_like(pipe_sorted_mm)
    out[inv] = pipe_sorted_mm
    return {
        "pipe_fault_mm": out,
        "soil_fault_mm": soil_fault_mm[:n],
        "s_nodes_offset_m": s[:n],
        "in_center_zone": zone[:n],
        "friction_cap_kN": np.asarray(friction_cap, dtype=float).reshape(-1)[:n],
        "tangent_xy": np.asarray(tangent_xy, dtype=float)[:n] if tangent_xy is not None else np.empty((0, 2), dtype=float),
        "normal_xy": np.asarray(normal_xy, dtype=float)[:n] if normal_xy is not None else np.empty((0, 2), dtype=float),
        "axial_iter_counts": axial_iter_counts,
        "axial_converged": axial_converged,
    }
