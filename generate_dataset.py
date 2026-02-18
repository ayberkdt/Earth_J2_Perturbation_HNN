# -*- coding: utf-8 -*-
"""
generate_dataset.py
===================
ODE (scipy.solve_ivp) ile rastgele başlangıç koşullarından Two-body + J2 yörünge verisi üretir
ve CSV + meta JSON olarak kaydeder.

CSV kolonları (train_pinn.py ile uyumluluk için KORUNUR):
t, x, y, z, vx, vy, vz, orbit_id, x0, y0, z0, vx0, vy0, vz0

Önemli not
----------
- t_eval her zaman [0, duration] aralığında kalacak şekilde üretilir (duration, dt'ye
  tam bölünmediğinde solve_ivp hatalarını önlemek için).
"""

# =======================================================================
# 0.                            IMPORTS
# =======================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from orbit_core import (
    DAY_S,
    MU_EARTH,
    R_EARTH,
    coe_to_eci_state,       
    two_body_j2_dynamics,    
    seed_everything,
)


# ============================================================================
# 1.                                 CSV schema 
# ============================================================================

# Do not change: training code depends on these
CSV_COLUMNS: List[str] = [
    "t", "x", "y", "z", "vx", "vy", "vz",
    "orbit_id",
    "x0", "y0", "z0", "vx0", "vy0", "vz0",
]

STATE_COLS: List[str] = ["x", "y", "z", "vx", "vy", "vz"]
IC_COLS: List[str] = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]


# ============================================================================
# 2.                    Utility: robust time grid
# ============================================================================

def make_time_grid(duration_s: float, dt_s: float, *, include_end: bool = True) -> np.ndarray:
    """
    Create a robust t_eval grid that is guaranteed to lie within [0, duration].

    Parameters
    ----------
    duration_s:
        Simulation duration [s]. Must be > 0.
    dt_s:
        Sampling step [s]. Must be > 0.
    include_end:
        If True, ensures duration is included as the last sample.

    Returns
    -------
    np.ndarray
        Monotonically increasing time array t_eval [s].
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0 (got {duration_s}).")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0 (got {dt_s}).")

    t = np.arange(0.0, duration_s, dt_s, dtype=np.float64)

    if include_end:
        if t.size == 0 or t[-1] < duration_s:
            t = np.append(t, np.float64(duration_s))
        else:
            # Rare: floating rounding may slightly overshoot; pin to duration
            t[-1] = np.float64(duration_s)

    # Safety: clip tiny rounding overshoots
    t = np.clip(t, 0.0, duration_s).astype(np.float64)

    # Safety: ensure strict monotonicity
    if t.size >= 2 and np.any(np.diff(t) <= 0):
        t = np.unique(t)
        if include_end and (t.size == 0 or t[-1] < duration_s):
            t = np.append(t, np.float64(duration_s))

    return t


# ============================================================================
# 3.                                 Config
# ============================================================================

@dataclass(slots=True)
class DatasetConfig:
    """
    Configuration for orbit dataset generation.

    Notes
    -----
    altitude_range_km is interpreted as *perigee altitude* range.
    """
    # Sampling ranges
    altitude_range_km: Tuple[float, float] = (300.0, 36_000.0)   # perigee altitude [km]
    eccentricity_range: Tuple[float, float] = (0.0, 0.2)         # [-]
    inclination_range_rad: Tuple[float, float] = (0.0, np.pi)    # [rad]

    # Simulation
    num_orbits: int = 1000
    sim_duration_s: float = 1.0 * DAY_S
    dt_sample_s: float = 600.0
    include_end: bool = True

    # Integrator
    ode_method: str = "DOP853"
    max_step_s: Optional[float] = None  # None -> solver default (recommended)
    rtol: float = 1e-9
    atol: float = 1e-9

    # Robustness
    max_attempts_multiplier: float = 3.0  # max_attempts = ceil(num_orbits * multiplier)

    # Output
    output_dir: str = "dataset"
    dataset_name: str = "orbit_j2_dataset_v2"  # keep default for compatibility
    seed: int = 42
    verbose: bool = True


# ============================================================================
# 4.                             Core generator
# ============================================================================

def _sanitize_dataset_name(name: str) -> str:
    name = str(name).strip()
    return name[:-4] if name.lower().endswith(".csv") else name


def generate_dataset(cfg: DatasetConfig) -> str:
    """
    Generate orbit dataset and write:
      - CSV:  <output_dir>/<dataset_name>.csv
      - JSON: <output_dir>/<dataset_name>_meta.json

    Returns
    -------
    str
        Path to the generated CSV.
    """
    cfg.dataset_name = _sanitize_dataset_name(cfg.dataset_name)

    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    t_eval_s = make_time_grid(cfg.sim_duration_s, cfg.dt_sample_s, include_end=cfg.include_end)
    n_t = int(t_eval_s.size)

    if cfg.verbose:
        print(
            "[generate_dataset] start | "
            f"target_orbits={cfg.num_orbits} | points/orbit={n_t} | "
            f"method={cfg.ode_method} | duration_s={cfg.sim_duration_s} | dt_s={cfg.dt_sample_s}"
        )

    blocks: List[np.ndarray] = []
    failures: List[Dict[str, str]] = []

    max_attempts = int(np.ceil(cfg.num_orbits * max(1.0, float(cfg.max_attempts_multiplier))))
    attempts = 0
    orbit_id = 0

    # Pre-alloc helpers
    t_span = (0.0, float(cfg.sim_duration_s))

    while orbit_id < cfg.num_orbits and attempts < max_attempts:
        attempts += 1

        # --- Random COE sample (perigee altitude interpreted) ---
        h_perigee_km = float(rng.uniform(*cfg.altitude_range_km))
        e = float(rng.uniform(*cfg.eccentricity_range))
        inc = float(rng.uniform(*cfg.inclination_range_rad))
        raan = float(rng.uniform(0.0, 2.0 * np.pi))
        argp = float(rng.uniform(0.0, 2.0 * np.pi))
        nu = float(rng.uniform(0.0, 2.0 * np.pi))

        rp_km = float(R_EARTH + h_perigee_km)
        if rp_km <= 0.0:
            failures.append({"attempt": str(attempts), "reason": "non_positive_radius", "detail": f"rp_km={rp_km}"})
            continue

        # a = rp / (1-e) for perigee radius rp = a(1-e)
        a_km = rp_km / (1.0 - e)

        y0_km_km_s = coe_to_eci_state(a_km, e, inc, raan, argp, nu, mu_km3_s2=MU_EARTH).astype(np.float64)

        # --- Integrate ---
        try:
            kwargs = dict(
                fun=two_body_j2_dynamics,
                t_span=t_span,
                y0=y0_km_km_s,
                t_eval=t_eval_s,
                method=str(cfg.ode_method),
                rtol=float(cfg.rtol),
                atol=float(cfg.atol),
            )
            # Only pass max_step if user specifies; let solver defaults be solver defaults.
            if cfg.max_step_s is not None:
                kwargs["max_step"] = float(cfg.max_step_s)

            sol = solve_ivp(**kwargs)
        except Exception as ex:
            failures.append({"attempt": str(attempts), "reason": "exception", "detail": repr(ex)})
            continue

        if (not sol.success) or (sol.y is None) or (sol.t is None):
            failures.append({"attempt": str(attempts), "reason": "integrator_failed", "detail": str(getattr(sol, "message", ""))})
            continue

        if sol.y.shape[1] != n_t:
            failures.append({"attempt": str(attempts), "reason": "bad_sample_count", "detail": f"n={sol.y.shape[1]} expected={n_t}"})
            continue

        # --- Assemble block ---
        orbit_ids = np.full(n_t, orbit_id, dtype=np.int32)
        y0_rep = np.tile(y0_km_km_s.reshape(1, 6), (n_t, 1))

        # block: [t, state(6), orbit_id, ic(6)]
        block = np.column_stack(
            (
                sol.t.astype(np.float64),
                sol.y.T.astype(np.float64),
                orbit_ids,
                y0_rep.astype(np.float64),
            )
        )
        blocks.append(block)

        orbit_id += 1
        if cfg.verbose and (orbit_id % 50 == 0):
            print(f"  .. ok_orbits={orbit_id}/{cfg.num_orbits} | attempts={attempts} | fails={len(failures)}")

    if orbit_id < cfg.num_orbits:
        raise RuntimeError(
            "Insufficient successful orbits. "
            f"ok={orbit_id}/{cfg.num_orbits} | attempts={attempts}/{max_attempts} | fails={len(failures)}. "
            "Consider widening ranges or loosening tolerances."
        )

    data = np.vstack(blocks)
    df = pd.DataFrame(data, columns=CSV_COLUMNS)

    csv_path = os.path.join(cfg.output_dir, f"{cfg.dataset_name}.csv")
    df.to_csv(csv_path, index=False)

    meta = asdict(cfg)
    meta.update(
        {
            "mu_km3_s2": MU_EARTH,
            "r_ref_km": R_EARTH,
            "gravity_model": "Two-body + J2",
            "columns": CSV_COLUMNS,
            "points_per_orbit": n_t,
            "successful_orbits": int(cfg.num_orbits),
            "attempts_total": int(attempts),
            "failures_count": int(len(failures)),
            "failures_sample": failures[:20],
        }
    )

    meta_path = os.path.join(cfg.output_dir, f"{cfg.dataset_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if cfg.verbose:
        print(f"[generate_dataset] OK | csv={csv_path}")
        print(f"[generate_dataset] OK | meta={meta_path}")

    return csv_path

