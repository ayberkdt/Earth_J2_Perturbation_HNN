# orbit_core.py
# -*- coding: utf-8 -*-
"""
orbit_core.py
=============
Ortak çekirdek modül: yörünge veri üretimi + PINN eğitimi/değerlendirmesi için
tek bir “source of truth”.

İçerik
------
- Fiziksel sabitler (km, s)
- Two-body + J2 yerçekimi dinamiği (ODE RHS)
- COE -> ECI kartesyen state dönüşümü
- Canonical (boyutsuz) ölçekleme (DU, TU, VU)
- PyTorch dataset sınıfları (operator learning / dynamics learning)
- PINN model bileşenleri (Fourier/Phase, MLP/SIREN, DeepONet)
- Physics loss (kinematik + dinamik + opsiyonel enerji tutarlılığı)
- Checkpoint (save/load) ve metrik yardımcıları

Önemli Notlar
-------------
- Tüm fiziksel büyüklükler aksi belirtilmedikçe (km, s) birimindedir.
- Canonical uzayda: DU=r_ref, TU=sqrt(DU^3/mu), VU=DU/TU -> mu'=1 ve r_ref'=1.
- ODE RHS fonksiyonları SciPy `solve_ivp` ile uyumludur: f(t, y) -> dy/dt.

Değişiklik Özeti
----------------
- ODE dinamiğinde daha güvenli “NaN guard”: r normu sonlu değilse veya ~0 ise
  NaN döndürerek solver’ın temiz şekilde fail etmesi sağlandı.
- Dataset tarafında (t, IC, state) dönüşümleri daha açık ve izlenebilir hale getirildi;
  IC ölçekleme adımı dummy time ile netleştirildi.
- Physics loss içinde sayısal güvenlik iyileştirmeleri (clamp_min, bound-orbit guard)
  ve enerji tutarlılığı opsiyonu daha kontrollü hale getirildi.
"""

# =======================================================================
# 0.                            IMPORTS
# =======================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, List

import os
import random

import math
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import Dataset



# =======================================================================
# 1.                        Fiziksel Sabitler (km, s)
# =======================================================================

MU_EARTH: float = 398_600.4418   # km^3/s^2
R_EARTH: float  = 6_378.137      # km
J2_EARTH: float = 1.082626e-3    # -
DAY_S: float    = 86_400.0       # s



# =======================================================================
# 2.                         Tekrarlanabilirlik
# =======================================================================

def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """
    Rastgelelik kaynaklarını tek bir noktadan sabitleyerek tekrarlanabilirlik sağlar.

    Bu fonksiyon sırasıyla şunları seed'ler:
      - Python `random`
      - NumPy
      - PyTorch (kuruluysa): CPU ve (varsa) CUDA RNG'leri

    Parametreler
    ------------
    seed:
        Tüm RNG kaynakları için kullanılacak global seed.
    deterministic:
        True ise, PyTorch tarafında mümkün olan yerlerde deterministik çalışma
        zorlanır (cuDNN ayarları + deterministic algorithms). Bu, bazı
        işlemlerde performansı düşürebilir veya deterministik kernel yoksa
        hata verebilir.

    Notlar
    ------
    - Tam determinism; GPU, sürücü ve PyTorch sürümüne göre değişebilir.
    - `CUBLAS_WORKSPACE_CONFIG` bazı CUDA matmul/GEMM işlemlerinde determinism
      için gereklidir; mümkünse programın başında çağırın.
    """
    # Python & NumPy
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch opsiyonel
    if torch is None:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDA GEMM determinism (MLP/Linear katmanları için faydalı)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # cuDNN determinism / benchmark
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)

    # Sessiz non-determinism yerine (mümkünse) açık hata
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Bazı sürümlerde/cihazlarda destek sınırlı olabilir
            pass


def get_device(prefer: str = "auto") -> "torch.device":
    """
    Eğitim/inference için en uygun cihazı seçer.

    Parametreler
    ------------
    prefer:
        "auto" -> CUDA varsa CUDA, yoksa MPS (Apple Silicon), yoksa CPU
        "cuda" -> CUDA yoksa hata
        "mps"  -> MPS yoksa hata
        "cpu"  -> her durumda CPU

    Döndürür
    --------
    torch.device
        Seçilen cihaz.
    """
    if torch is None:
        raise RuntimeError("PyTorch bulunamadı. Model eğitimi/inference için torch gerekli.")

    prefer = prefer.strip().lower()

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer in ("cuda", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "cuda":
            raise RuntimeError("CUDA istendi fakat bu sistemde CUDA kullanılabilir değil.")

    if prefer in ("mps", "auto"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "mps":
            raise RuntimeError("MPS istendi fakat bu sistemde MPS kullanılabilir değil.")

    return torch.device("cpu")



# ============================================================================
# 3.                  ODE Dynamics: Two-Body + J2 (Earth)
# ============================================================================

def two_body_j2_dynamics(
    t: float,
    y: np.ndarray,
    mu_km3_s2: float = MU_EARTH,
    r_ref_km: float = R_EARTH,
    j2: float = J2_EARTH,
) -> np.ndarray:
    """
    Earth gravity dynamics with J2 perturbation (ECI frame).

    Parameters
    ----------
    t:
        Time [s]. (Not used explicitly; included for ODE solver signature.)
    y:
        State vector [x, y, z, vx, vy, vz] with units:
        - position: km
        - velocity: km/s
    mu_km3_s2:
        Gravitational parameter μ [km^3/s^2].
    r_ref_km:
        Reference radius (typically Earth's equatorial radius) [km].
    j2:
        Second zonal harmonic coefficient J2 [-].

    Returns
    -------
    np.ndarray
        Time derivative dy/dt = [vx, vy, vz, ax, ay, az] (km/s, km/s^2).

    Notes
    -----
    If the position norm is non-finite or nearly zero, NaNs are returned to
    force a clean failure in `solve_ivp`.
    """
    y = np.asarray(y, dtype=np.float64)

    r_eci_km = y[:3]
    v_eci_km_s = y[3:]

    r_km = float(np.linalg.norm(r_eci_km))
    if (not np.isfinite(r_km)) or (r_km < 1e-12):
        return np.full_like(y, np.nan, dtype=np.float64)

    # --- Two-body (monopole) acceleration ---
    a_tb_km_s2 = -mu_km3_s2 * r_eci_km / (r_km**3)

    # --- J2 perturbation ---
    z_over_r = r_eci_km[2] / r_km
    z2_over_r2 = z_over_r * z_over_r

    j2_coeff = 1.5 * j2 * mu_km3_s2 * (r_ref_km**2) / (r_km**5)
    common_xy = (5.0 * z2_over_r2 - 1.0)

    a_j2_km_s2 = j2_coeff * np.array(
        [
            r_eci_km[0] * common_xy,
            r_eci_km[1] * common_xy,
            r_eci_km[2] * (5.0 * z2_over_r2 - 3.0),
        ],
        dtype=np.float64,
    )

    a_eci_km_s2 = a_tb_km_s2 + a_j2_km_s2
    return np.hstack((v_eci_km_s, a_eci_km_s2)).astype(np.float64)



# ============================================================================
# 4.       Classical Orbital Elements (COE) -> Cartesian State (ECI)
# ============================================================================

def coe_to_eci_state(
    a_km: float,
    e: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    nu_rad: float,
    mu_km3_s2: float = MU_EARTH,
) -> np.ndarray:
    """
    Convert classical orbital elements (COE) to an ECI Cartesian state.

    Parameters
    ----------
    a_km:
        Semi-major axis [km].
    e:
        Eccentricity [-].
    inc_rad:
        Inclination i [rad].
    raan_rad:
        Right ascension of ascending node Ω [rad].
    argp_rad:
        Argument of perigee ω [rad].
    nu_rad:
        True anomaly ν [rad].
    mu_km3_s2:
        Gravitational parameter μ [km^3/s^2].

    Returns
    -------
    np.ndarray
        State vector [x, y, z, vx, vy, vz] in ECI with units:
        - position: km
        - velocity: km/s
    """
    # Semi-latus rectum
    p_km = a_km * (1.0 - e * e)

    cos_nu, sin_nu = np.cos(nu_rad), np.sin(nu_rad)
    denom = 1.0 + e * cos_nu

    r_pqw_km = np.array(
        [p_km * cos_nu / denom, p_km * sin_nu / denom, 0.0],
        dtype=np.float64,
    )

    v_scale = np.sqrt(mu_km3_s2 / p_km)
    v_pqw_km_s = np.array(
        [-v_scale * sin_nu, v_scale * (e + cos_nu), 0.0],
        dtype=np.float64,
    )

    cos_O, sin_O = np.cos(raan_rad), np.sin(raan_rad)
    cos_w, sin_w = np.cos(argp_rad), np.sin(argp_rad)
    cos_i, sin_i = np.cos(inc_rad), np.sin(inc_rad)

    # PQW -> ECI rotation matrix (3-1-3 sequence: Ω, i, ω)
    C_eci_pqw = np.array(
        [
            [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i,  sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
            [sin_w * sin_i,                           cos_w * sin_i,                           cos_i],
        ],
        dtype=np.float64,
    )

    r_eci_km = C_eci_pqw @ r_pqw_km
    v_eci_km_s = C_eci_pqw @ v_pqw_km_s
    return np.hstack((r_eci_km, v_eci_km_s)).astype(np.float64)



# ============================================================================
# 5.                     Canonical (Boyutsuz) Ölçekleme
# ============================================================================

@dataclass(frozen=True, slots=True)
class CanonicalScaleConfig:
    """
    Configuration for canonical (non-dimensional) scaling.

    Parameters
    ----------
    r_ref_km:
        Reference length scale [km]. Common choice: Earth's equatorial radius.
    mu_km3_s2:
        Gravitational parameter μ [km^3/s^2].
    """
    r_ref_km: float = R_EARTH
    mu_km3_s2: float = MU_EARTH


class CanonicalScaler:
    """
    Canonical scaling for orbital dynamics (non-dimensionalization).

    Definitions
    -----------
    DU = r_ref_km
    TU = sqrt(DU^3 / μ)
    VU = DU / TU

    With this choice, the non-dimensional gravitational parameter becomes μ' = 1.

    Notes
    -----
    - `transform` expects time as (N,) or (N,1) and state as (N,6) or (6,).
    - The returned arrays are always shaped as:
        t_nd: (N,1)
        y_nd: (N,6)
    """

    def __init__(self, cfg: CanonicalScaleConfig = CanonicalScaleConfig()):
        self.cfg = cfg

        self.DU_km: float = float(cfg.r_ref_km)
        self.TU_s: float = float(np.sqrt((cfg.r_ref_km ** 3) / cfg.mu_km3_s2))
        self.VU_km_s: float = self.DU_km / self.TU_s

    def transform(self, t_s: np.ndarray, y_km_km_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert physical units -> canonical (non-dimensional) units.

        Parameters
        ----------
        t_s:
            Time array [s], shape (N,) or (N,1).
        y_km_km_s:
            State array [km, km/s], shape (N,6) or (6,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            t_nd:
                Non-dimensional time, shape (N,1).
            y_nd:
                Non-dimensional state [r/DU, v/VU], shape (N,6).
        """
        t_s_arr = np.asarray(t_s, dtype=np.float64).reshape(-1, 1)
        y_arr = np.asarray(y_km_km_s, dtype=np.float64).reshape(-1, 6)

        t_nd = t_s_arr / self.TU_s
        r_nd = y_arr[:, :3] / self.DU_km
        v_nd = y_arr[:, 3:] / self.VU_km_s

        return t_nd.astype(np.float64), np.hstack((r_nd, v_nd)).astype(np.float64)

    def inverse_transform(self, t_nd: np.ndarray, y_nd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert canonical (non-dimensional) units -> physical units.

        Parameters
        ----------
        t_nd:
            Non-dimensional time, shape (N,) or (N,1).
        y_nd:
            Non-dimensional state, shape (N,6) or (6,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            t_s:
                Time [s], shape (N,1).
            y_km_km_s:
                State [km, km/s], shape (N,6).
        """
        t_nd_arr = np.asarray(t_nd, dtype=np.float64).reshape(-1, 1)
        y_nd_arr = np.asarray(y_nd, dtype=np.float64).reshape(-1, 6)

        t_s = t_nd_arr * self.TU_s
        r_km = y_nd_arr[:, :3] * self.DU_km
        v_km_s = y_nd_arr[:, 3:] * self.VU_km_s

        return t_s.astype(np.float64), np.hstack((r_km, v_km_s)).astype(np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize scaler parameters for logging / checkpoint metadata.
        """
        return {
            "cfg": asdict(self.cfg),
            "DU_km": self.DU_km,
            "TU_s": self.TU_s,
            "VU_km_s": self.VU_km_s,
        }



# ============================================================================
# 6) CSV Datasets
#    (a) Supervised operator learning: (t, IC) -> state(t)
#    (b) Dynamics learning: state -> dstate/dt  (canonical, analytic J2 model)
# ============================================================================

DEFAULT_STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
DEFAULT_IC_COLS = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]

class OrbitOperatorDataset(Dataset):
    """
    Supervised dataset for operator learning: (t, initial_state) -> state(t).

    The CSV is expected to contain:
      - time column: `t_col` (seconds)
      - initial condition columns: `ic_cols` (km, km/s)
      - state columns at time t: `state_cols` (km, km/s)

    All values are converted to canonical (non-dimensional) units using `scaler`.

    Returns (per sample)
    --------------------
    t_nd : (1,) or (1,1) tensor
        Non-dimensional time.
    ic_nd : (6,) tensor
        Non-dimensional initial condition (r/DU, v/VU).
    y_nd : (6,) tensor
        Non-dimensional state at time t (r/DU, v/VU).
    """

    def __init__(
        self,
        csv_path: str,
        scaler: CanonicalScaler,
        *,
        t_col: str = "t",
        state_cols: Optional[List[str]] = None,
        ic_cols: Optional[List[str]] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("OrbitOperatorDataset requires PyTorch.")

        state_cols = state_cols or DEFAULT_STATE_COLS
        ic_cols = ic_cols or DEFAULT_IC_COLS
        dtype = dtype or torch.float32

        df = pd.read_csv(csv_path)

        # --- Validate required columns early (clear error messages) ---
        required = [t_col, *state_cols, *ic_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        t_s = df[[t_col]].to_numpy(dtype=np.float64)             # (N,1) [s]
        y_phys = df[state_cols].to_numpy(dtype=np.float64)       # (N,6) [km, km/s]
        ic_phys = df[ic_cols].to_numpy(dtype=np.float64)         # (N,6) [km, km/s]

        # state(t): physical -> canonical
        t_nd, y_nd = scaler.transform(t_s, y_phys)

        # IC is time-independent; scaling velocities needs VU only, but we use
        # the same scaler interface with a dummy time vector for clarity.
        t_dummy = np.zeros_like(t_s, dtype=np.float64)
        _, ic_nd = scaler.transform(t_dummy, ic_phys)

        self.t_nd = torch.as_tensor(t_nd, dtype=dtype)   # (N,1)
        self.ic_nd = torch.as_tensor(ic_nd, dtype=dtype) # (N,6)
        self.y_nd = torch.as_tensor(y_nd, dtype=dtype)   # (N,6)

    def __len__(self) -> int:
        return int(self.t_nd.shape[0])

    def __getitem__(self, idx: int):
        return self.t_nd[idx], self.ic_nd[idx], self.y_nd[idx]

# ----------------------------------------------------------------------------
# Canonical dynamics helpers (mu=1, r_ref=1) for J2-perturbed gravity
# ----------------------------------------------------------------------------

def accel_j2_canonical_torch(r_nd: "torch.Tensor", *, j2: float = J2_EARTH) -> "torch.Tensor":
    """
    Canonical (non-dimensional) acceleration including J2 perturbation.

    Assumptions (canonical units)
    -----------------------------
    - mu = 1
    - r_ref = 1
    - r_nd is non-dimensional

    Parameters
    ----------
    r_nd:
        Position tensor in canonical units, shape (N,3).
    j2:
        J2 coefficient (dimensionless).

    Returns
    -------
    torch.Tensor
        Acceleration tensor in canonical units, shape (N,3).
    """
    r_norm = torch.norm(r_nd, dim=1, keepdim=True).clamp_min(1e-12)

    # Two-body (monopole)
    a_tb = -r_nd / (r_norm ** 3)

    # J2 perturbation (canonical)
    x, y, z = r_nd[:, 0:1], r_nd[:, 1:2], r_nd[:, 2:3]
    z2_over_r2 = (z / r_norm).pow(2)

    coeff = 1.5 * float(j2) / (r_norm ** 5)
    common_xy = (5.0 * z2_over_r2 - 1.0)

    a_j2 = torch.cat(
        [
            coeff * x * common_xy,
            coeff * y * common_xy,
            coeff * z * (5.0 * z2_over_r2 - 3.0),
        ],
        dim=1,
    )
    return a_tb + a_j2


def potential_j2_canonical_torch(r_nd: "torch.Tensor", *, j2: float = J2_EARTH) -> "torch.Tensor":
    """
    Canonical gravitational potential: U = U_two_body + U_J2.

    Canonical units (mu=1, r_ref=1):
      U_two_body = -1 / r
      U_J2       = (J2/2) * (1/r^3) * (3*(z/r)^2 - 1)

    Parameters
    ----------
    r_nd:
        Position tensor in canonical units, shape (N,3).
    j2:
        J2 coefficient (dimensionless).

    Returns
    -------
    torch.Tensor
        Potential U in canonical units, shape (N,1).
    """
    r_norm = torch.norm(r_nd, dim=1, keepdim=True).clamp_min(1e-12)
    z = r_nd[:, 2:3]
    z2_over_r2 = (z / r_norm).pow(2)

    u_tb = -1.0 / r_norm
    u_j2 = 0.5 * float(j2) * (1.0 / (r_norm ** 3)) * (3.0 * z2_over_r2 - 1.0)
    return u_tb + u_j2


def hamiltonian_canonical_torch(state_nd: "torch.Tensor", *, j2: float = J2_EARTH) -> "torch.Tensor":
    """
    Canonical Hamiltonian H = T + U for the J2-perturbed gravity model.

    Parameters
    ----------
    state_nd:
        Canonical state tensor, shape (N,6) = [r(3), v(3)].
    j2:
        J2 coefficient (dimensionless).

    Returns
    -------
    torch.Tensor
        Hamiltonian values, shape (N,1).
    """
    r_nd = state_nd[:, 0:3]
    v_nd = state_nd[:, 3:6]

    kinetic = 0.5 * torch.sum(v_nd * v_nd, dim=1, keepdim=True)
    potential = potential_j2_canonical_torch(r_nd, j2=float(j2))
    return kinetic + potential


class OrbitDynamicsDataset(Dataset):
    """
    Dataset for dynamics learning (HNN / Neural ODE style): state -> dstate/dt.

    The CSV provides state samples in physical units (km, km/s). We:
      1) Convert states to canonical units via `scaler` (time is irrelevant here).
      2) Compute analytic canonical derivatives under (mu=1, J2) model:
           dr/dt = v
           dv/dt = a(r)

    Returns (per sample)
    --------------------
    y_nd    : (6,) tensor   canonical state
    dydt_nd : (6,) tensor   canonical time derivative
    """

    def __init__(
        self,
        csv_path: str,
        scaler: CanonicalScaler,
        *,
        state_cols: Optional[List[str]] = None,
        dtype: Optional["torch.dtype"] = None,
        j2: float = J2_EARTH,
    ) -> None:
        if torch is None:
            raise RuntimeError("OrbitDynamicsDataset requires PyTorch.")

        state_cols = state_cols or DEFAULT_STATE_COLS
        dtype = dtype or torch.float32

        df = pd.read_csv(csv_path)

        missing = [c for c in state_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required state columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        y_phys = df[state_cols].to_numpy(dtype=np.float64)  # (N,6) [km, km/s]

        # Convert to canonical using dummy time (clarity + consistent interface)
        t_dummy = np.zeros((y_phys.shape[0], 1), dtype=np.float64)
        _, y_nd = scaler.transform(t_dummy, y_phys)

        self.y_nd = torch.as_tensor(y_nd, dtype=dtype)  # (N,6)

        r_nd = self.y_nd[:, 0:3]
        v_nd = self.y_nd[:, 3:6]
        a_nd = accel_j2_canonical_torch(r_nd, j2=float(j2))

        self.dydt_nd = torch.cat([v_nd, a_nd], dim=1)  # (N,6)

    def __len__(self) -> int:
        return int(self.y_nd.shape[0])

    def __getitem__(self, idx: int):
        return self.y_nd[idx], self.dydt_nd[idx]


# ============================================================================
# 7) PINN Model Building Blocks
#    - Time feature encoders (Fourier / Phase)
#    - Backbones (Tanh MLP / SIREN)
#    - Operator heads (DeepONet)
#    - Optional HNN utilities
# ============================================================================


# ============================================================================
# 7.1) Time encoders
# ============================================================================

class FourierFeatures(nn.Module):
    """
    Deterministic Fourier feature encoder (positional encoding) for time input.

    Motivation
    ----------
    Standard MLPs tend to exhibit spectral bias (favor low frequencies). Fourier features
    help represent periodic / high-frequency components more easily.

    Output
    ------
    If include_input=True:
        [t, sin(2π f1 t), ..., sin(2π fK t), cos(2π f1 t), ..., cos(2π fK t)]
    else:
        [sin(...), cos(...)]
    """

    def __init__(
        self,
        num_frequencies: int = 16,
        min_frequency: float = 0.01,
        max_frequency: float = 10.0,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
        super().__init__()

        self.num_frequencies = int(num_frequencies)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.include_input = bool(include_input)
        self.log_sampling = bool(log_sampling)

        if self.num_frequencies < 0:
            raise ValueError("num_frequencies must be >= 0")
        if self.num_frequencies == 0:
            self.register_buffer("freq_bands", torch.zeros(0, dtype=torch.float32))
            return

        if self.min_frequency <= 0.0:
            raise ValueError("min_frequency must be > 0")
        if self.max_frequency <= self.min_frequency:
            raise ValueError("max_frequency must be > min_frequency")

        if self.num_frequencies == 1:
            bands = torch.tensor([self.max_frequency], dtype=torch.float32)
        else:
            if self.log_sampling:
                bands = torch.logspace(
                    math.log10(self.min_frequency),
                    math.log10(self.max_frequency),
                    steps=self.num_frequencies,
                    dtype=torch.float32,
                )
            else:
                bands = torch.linspace(
                    self.min_frequency,
                    self.max_frequency,
                    steps=self.num_frequencies,
                    dtype=torch.float32,
                )

        self.register_buffer("freq_bands", bands)

    def out_dim(self) -> int:
        base = 1 if self.include_input else 0
        return base + 2 * self.num_frequencies

    def forward(self, t: "torch.Tensor") -> "torch.Tensor":
        """
        Parameters
        ----------
        t : torch.Tensor
            Shape (N, 1)

        Returns
        -------
        torch.Tensor
            Shape (N, out_dim)
        """
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"FourierFeatures expects t shaped (N,1); got {tuple(t.shape)}")

        if self.num_frequencies == 0:
            return t if self.include_input else t.new_zeros((t.shape[0], 0))

        # keep dtype/device stable for mixed precision / float64 runs
        bands = self.freq_bands.to(device=t.device, dtype=t.dtype).view(1, -1)  # (1, K)
        angles = (2.0 * math.pi) * t * bands  # (N, K)

        s = torch.sin(angles)
        c = torch.cos(angles)

        if self.include_input:
            return torch.cat([t, s, c], dim=1)
        return torch.cat([s, c], dim=1)


def _mean_motion_from_ic_canonical(ic: "torch.Tensor") -> "torch.Tensor":
    """
    Compute mean motion n from initial condition (canonical units, mu=1).

    Parameters
    ----------
    ic : torch.Tensor
        Shape (N, 6) where ic = [r0(3), v0(3)] in canonical units.

    Returns
    -------
    torch.Tensor
        Mean motion n, shape (N, 1). (Effectively rad / TU in canonical time.)
    """
    if ic.ndim != 2 or ic.shape[1] != 6:
        raise ValueError(f"Expected ic shaped (N,6); got {tuple(ic.shape)}")

    r0 = ic[:, 0:3]
    v0 = ic[:, 3:6]

    r = torch.norm(r0, dim=1, keepdim=True).clamp_min(1e-12)
    v2 = torch.sum(v0 * v0, dim=1, keepdim=True)

    # Specific orbital energy (canonical mu=1)
    eps = 0.5 * v2 - 1.0 / r

    # Only bound orbits are intended here; guard against eps >= 0
    eps = torch.clamp(eps, max=-1e-8)

    a = (-1.0 / (2.0 * eps)).clamp_min(1e-6)  # semi-major axis (canonical DU)
    n = torch.sqrt(1.0 / (a**3))              # sqrt(mu/a^3), mu=1

    return n


class PhaseFeatures(nn.Module):
    """
    Period-aware time features using phase = n0 * t, where n0 is computed from IC.

    Motivation
    ----------
    A single network learning a wide range of periods (LEO -> GEO) can struggle when
    time is fed directly. Using phase aligns trajectories by their natural frequency.

    Output
    ------
    Optionally includes:
      - t
      - phase
    plus harmonics:
      sin(k*phase), cos(k*phase), k=1..K
    """

    def __init__(
        self,
        num_harmonics: int = 8,
        include_t: bool = True,
        include_phase: bool = False,
    ) -> None:
        super().__init__()
        self.num_harmonics = int(num_harmonics)
        self.include_t = bool(include_t)
        self.include_phase = bool(include_phase)

        if self.num_harmonics < 0:
            raise ValueError("num_harmonics must be >= 0")

    def out_dim(self) -> int:
        base = (1 if self.include_t else 0) + (1 if self.include_phase else 0)
        return base + 2 * self.num_harmonics

    def forward(self, t: "torch.Tensor", ic: "torch.Tensor") -> "torch.Tensor":
        """
        Parameters
        ----------
        t : torch.Tensor
            Shape (N, 1)
        ic : torch.Tensor
            Shape (N, 6) in canonical units

        Returns
        -------
        torch.Tensor
            Shape (N, out_dim)
        """
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"PhaseFeatures expects t shaped (N,1); got {tuple(t.shape)}")
        if ic.ndim != 2 or ic.shape[1] != 6:
            raise ValueError(f"PhaseFeatures expects ic shaped (N,6); got {tuple(ic.shape)}")

        n0 = _mean_motion_from_ic_canonical(ic)  # (N,1)
        phase = n0 * t                           # (N,1)

        feats: List["torch.Tensor"] = []
        if self.include_t:
            feats.append(t)
        if self.include_phase:
            feats.append(phase)

        if self.num_harmonics > 0:
            k = torch.arange(
                1, self.num_harmonics + 1,
                device=t.device,
                dtype=t.dtype
            ).view(1, -1)  # (1, K)
            ang = phase * k  # (N, K)
            feats.append(torch.sin(ang))
            feats.append(torch.cos(ang))

        if not feats:
            return t.new_zeros((t.shape[0], 0))
        return torch.cat(feats, dim=1)


# ============================================================================
# 7.2) Backbones: SIREN / Tanh MLP
# ============================================================================

class SineLayer(nn.Module):
    """SIREN sine layer (Sitzmann et al.) with paper-style initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        w0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.w0 = float(w0)

        self.linear = nn.Linear(self.in_features, int(out_features), bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.w0

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.sin(self.w0 * self.linear(x))


class SirenNet(nn.Module):
    """SIREN MLP: stacked sine layers + linear head."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 6,
        *,
        hidden: int = 128,
        depth: int = 3,
        w0_initial: float = 30.0,
        w0: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        in_dim = int(in_dim)
        out_dim = int(out_dim)
        hidden = int(hidden)
        depth = int(depth)

        if depth < 1:
            raise ValueError("SirenNet depth must be >= 1")

        layers: List[nn.Module] = [SineLayer(in_dim, hidden, bias=bias, is_first=True, w0=w0_initial)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden, hidden, bias=bias, is_first=False, w0=w0))

        head = nn.Linear(hidden, out_dim, bias=bias)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / w0
            head.weight.uniform_(-bound, bound)
            if head.bias is not None:
                head.bias.uniform_(-bound, bound)

        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


def _make_tanh_mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> "nn.Module":
    """Simple Tanh MLP builder."""
    in_dim = int(in_dim)
    out_dim = int(out_dim)
    hidden = int(hidden)
    depth = int(depth)

    if depth < 1:
        raise ValueError("MLP depth must be >= 1")

    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


# ============================================================================
# 7.3) Operator head: DeepONet
# ============================================================================

class DeepONetCore(nn.Module):
    """
    Parametric operator: IC -> y(t) using DeepONet factorization.

    trunk(t_feat)   : (N, p)
    branch(ic_feat) : (N, out_dim*p) -> reshape (N, out_dim, p)
    out_j(t)        : sum_k branch_{j,k}(ic) * trunk_k(t) + bias_j
    """

    def __init__(
        self,
        trunk_in: int,
        branch_in: int,
        *,
        out_dim: int = 6,
        latent: int = 64,
        hidden: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.latent = int(latent)

        self.trunk = _make_tanh_mlp(trunk_in, self.latent, hidden, depth)
        self.branch = _make_tanh_mlp(branch_in, self.out_dim * self.latent, hidden, depth)
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, t_feat: "torch.Tensor", ic_feat: "torch.Tensor") -> "torch.Tensor":
        trunk = self.trunk(t_feat)  # (N, p)
        branch = self.branch(ic_feat).view(-1, self.out_dim, self.latent)  # (N, out, p)
        return (branch * trunk.unsqueeze(1)).sum(dim=-1) + self.bias


# ============================================================================
# 7.4) Optional: Hamiltonian NN utilities
# ============================================================================

class HamiltonianNN(nn.Module):
    """Hamiltonian Neural Network (HNN): state -> H(state)."""

    def __init__(self, *, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.net = _make_tanh_mlp(6, 1, hidden, depth)

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        return self.net(state)


def hnn_vector_field(hnn: HamiltonianNN, state: "torch.Tensor") -> "torch.Tensor":
    """
    Vector field induced by HNN: dstate/dt = J ∇H, with canonical (q,p) = (r,v).

    J = [[0, I], [-I, 0]]
    """
    x = state.clone().detach().requires_grad_(True)
    H = hnn(x)  # (N,1)

    grad = torch.autograd.grad(
        H, x,
        grad_outputs=torch.ones_like(H),
        create_graph=True,
    )[0]  # (N,6)

    dH_dq = grad[:, 0:3]
    dH_dp = grad[:, 3:6]
    dqdt = dH_dp
    dpdt = -dH_dq
    return torch.cat([dqdt, dpdt], dim=1)


# ============================================================================
# 7.5) OrbitPINN: (t, IC) -> state(t) in canonical units
# ============================================================================

class OrbitPINN(nn.Module):
    """
    Orbit operator model: state(t, IC) -> [x, y, z, vx, vy, vz] (canonical).

    Supported architectures
    -----------------------
    Classical:
      - "mlp"              : concat([t, ic]) -> Tanh MLP
      - "fourier_mlp"      : concat([Fourier(t), ic]) -> Tanh MLP
      - "siren"            : concat([t, ic]) -> SIREN
      - "fourier_siren"    : concat([Fourier(t), ic]) -> SIREN

    Period-aware:
      - "phase_mlp"        : concat([Phase(t,ic), ic]) -> Tanh MLP
      - "phase_siren"      : concat([Phase(t,ic), ic]) -> SIREN

    Deep operator (recommended for IC -> trajectory operators):
      - "deeponet"         : DeepONet with raw t
      - "deeponet_fourier" : DeepONet with Fourier(t)
      - "deeponet_phase"   : DeepONet with Phase(t,ic)
    """

    _ALLOWED_ARCH = {
        "mlp", "fourier_mlp", "siren", "fourier_siren",
        "phase_mlp", "phase_siren",
        "deeponet", "deeponet_fourier", "deeponet_phase",
    }

    def __init__(
        self,
        *,
        hidden: int = 128,
        depth: int = 3,
        j2: float = J2_EARTH,
        arch: str = "deeponet_phase",
        # Fourier(t)
        fourier_features: int = 16,
        fourier_min_freq: float = 0.01,
        fourier_max_freq: float = 10.0,
        fourier_include_input: bool = True,
        fourier_log_sampling: bool = True,
        # Phase(t, ic)
        phase_harmonics: int = 8,
        phase_include_t: bool = True,
        phase_include_phase: bool = False,
        # SIREN
        siren_w0_initial: float = 30.0,
        siren_w0: float = 1.0,
        # DeepONet
        deeponet_latent: int = 64,
        deeponet_branch_aug: bool = True,
        # Hard constraint: y(t) = y0 + t * NN(...)
        hard_constraint: bool = False,
    ) -> None:
        super().__init__()

        self.j2 = float(j2)
        self.hard_constraint = bool(hard_constraint)

        arch = str(arch).lower().strip()
        if arch not in self._ALLOWED_ARCH:
            raise ValueError(f"Unsupported arch='{arch}'. Options: {sorted(self._ALLOWED_ARCH)}")
        self.arch = arch

        # ---- Mode flags ----
        self.is_deeponet: bool = arch.startswith("deeponet")
        self.uses_siren: bool = arch.endswith("siren")          # only for non-deeponet modes
        self.uses_fourier: bool = ("fourier" in arch)
        self.uses_phase: bool = arch.startswith("phase_") or arch.endswith("_phase")

        # ---- Time encoder ----
        self.ff: Optional[FourierFeatures] = None
        self.pf: Optional[PhaseFeatures] = None

        if self.uses_phase:
            self.pf = PhaseFeatures(
                num_harmonics=phase_harmonics,
                include_t=phase_include_t,
                include_phase=phase_include_phase,
            )
            time_feat_dim = self.pf.out_dim()
        elif self.uses_fourier:
            self.ff = FourierFeatures(
                num_frequencies=fourier_features,
                min_frequency=fourier_min_freq,
                max_frequency=fourier_max_freq,
                include_input=fourier_include_input,
                log_sampling=fourier_log_sampling,
            )
            time_feat_dim = self.ff.out_dim()
        else:
            time_feat_dim = 1

        # ---- Branch (IC) feature augmentation ----
        # [ic, |r0|, |v0|] improves generalization for wide-orbit families.
        self.branch_aug: bool = bool(deeponet_branch_aug)
        branch_feat_dim = 6 + (2 if self.branch_aug else 0)

        # ---- Core model ----
        self.net: Optional[nn.Module] = None
        self.deeponet: Optional[DeepONetCore] = None

        if self.is_deeponet:
            self.deeponet = DeepONetCore(
                trunk_in=time_feat_dim,
                branch_in=branch_feat_dim,
                out_dim=6,
                latent=deeponet_latent,
                hidden=hidden,
                depth=depth,
            )
        else:
            in_dim = time_feat_dim + 6  # time features + raw IC
            if self.uses_siren:
                self.net = SirenNet(
                    in_dim=in_dim,
                    out_dim=6,
                    hidden=hidden,
                    depth=depth,
                    w0_initial=siren_w0_initial,
                    w0=siren_w0,
                )
            else:
                self.net = _make_tanh_mlp(in_dim, 6, hidden, depth)

    # -------------------------
    # Feature construction
    # -------------------------

    def _time_features(self, t: "torch.Tensor", ic: "torch.Tensor") -> "torch.Tensor":
        if self.pf is not None:
            return self.pf(t, ic)
        if self.ff is not None:
            return self.ff(t)
        return t

    def _branch_features(self, ic: "torch.Tensor") -> "torch.Tensor":
        if not self.branch_aug:
            return ic
        r0 = ic[:, 0:3]
        v0 = ic[:, 3:6]
        r0_norm = torch.norm(r0, dim=1, keepdim=True)
        v0_norm = torch.norm(v0, dim=1, keepdim=True)
        return torch.cat([ic, r0_norm, v0_norm], dim=1)

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, t: "torch.Tensor", ic: "torch.Tensor") -> "torch.Tensor":
        """
        Parameters
        ----------
        t : torch.Tensor
            Shape (N,1), canonical time
        ic : torch.Tensor
            Shape (N,6), canonical initial state

        Returns
        -------
        torch.Tensor
            Shape (N,6), canonical state prediction
        """
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"OrbitPINN expects t shaped (N,1); got {tuple(t.shape)}")
        if ic.ndim != 2 or ic.shape[1] != 6:
            raise ValueError(f"OrbitPINN expects ic shaped (N,6); got {tuple(ic.shape)}")

        t_feat = self._time_features(t, ic)

        if self.is_deeponet:
            if self.deeponet is None:
                raise RuntimeError("Internal error: deeponet is not initialized.")
            ic_feat = self._branch_features(ic)
            core_out = self.deeponet(t_feat, ic_feat)
        else:
            if self.net is None:
                raise RuntimeError("Internal error: net is not initialized.")
            x = torch.cat([t_feat, ic], dim=1)
            core_out = self.net(x)

        # Hard constraint: enforce y(t=0)=ic
        if self.hard_constraint:
            return ic + t * core_out

        return core_out



# ============================================================================
# 8) Physics loss (PINN core)
# ============================================================================

def time_derivative(y: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
    """
    Compute element-wise time derivative dy/dt using autograd.

    Parameters
    ----------
    y:
        Tensor to differentiate, shape (N, D).
    t:
        Time tensor, shape (N, 1). Must have requires_grad=True (leaf preferred).

    Returns
    -------
    torch.Tensor
        dy/dt with shape (N, D).

    Notes
    -----
    This implementation computes D separate autograd.grad calls. It is robust and
    version-agnostic. Faster alternatives exist (e.g., torch.func + vmap/jacrev)
    but may be version-dependent.
    """
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"time_derivative expects t shaped (N,1); got {tuple(t.shape)}")
    if y.ndim != 2:
        raise ValueError(f"time_derivative expects y shaped (N,D); got {tuple(y.shape)}")

    grads: List["torch.Tensor"] = []
    ones = torch.ones_like(t)

    for k in range(y.shape[1]):
        g = torch.autograd.grad(
            y[:, k:k + 1],
            t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(g)

    return torch.cat(grads, dim=1)


def accel_j2_canonical_from_state(
    state_nd: "torch.Tensor",
    *,
    j2: float,
) -> "torch.Tensor":
    """
    Compute canonical acceleration a(r) = a_two_body + a_J2 from canonical state.

    Parameters
    ----------
    state_nd:
        Canonical state tensor, shape (N,6) = [r(3), v(3)].
    j2:
        J2 coefficient.

    Returns
    -------
    torch.Tensor
        Acceleration tensor, shape (N,3) in canonical units.
    """
    r = state_nd[:, 0:3]
    r_norm = torch.norm(r, dim=1, keepdim=True).clamp_min(1e-12)

    a_tb = -r / (r_norm ** 3)

    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    z2_over_r2 = (z / r_norm).pow(2)

    coeff = 1.5 * float(j2) / (r_norm ** 5)
    common_xy = (5.0 * z2_over_r2 - 1.0)

    a_j2 = torch.cat(
        [
            coeff * x * common_xy,
            coeff * y * common_xy,
            coeff * z * (5.0 * z2_over_r2 - 3.0),
        ],
        dim=1,
    )
    return a_tb + a_j2


def physics_loss(
    model: OrbitPINN,
    t: "torch.Tensor",
    ic: "torch.Tensor",
    *,
    state_nd: Optional["torch.Tensor"] = None,
    drdt_nd: Optional["torch.Tensor"] = None,
    dvdt_nd: Optional["torch.Tensor"] = None,
    energy_weight: float = 1.0,
    return_parts: bool = False,
) -> "torch.Tensor":
    """
    Physics residual loss in canonical units.

    Constraints enforced (canonical, mu=1, r_ref=1)
    ----------------------------------------------
      1) Kinematics:   dr/dt = v
      2) Dynamics:     dv/dt = a(r)  (two-body + J2)
      3) (Optional) Energy consistency: H(t) ≈ H(t0)  [soft constraint]

    Parameters
    ----------
    model:
        OrbitPINN model producing canonical states.
    t:
        Canonical time, shape (N,1). If requires_grad is False, a grad-enabled
        copy is created.
    ic:
        Canonical initial state, shape (N,6) = [r0, v0].
    state_nd:
        Optional precomputed model output, shape (N,6). If None, computed as model(t, ic).
    drdt_nd / dvdt_nd:
        Optional precomputed time derivatives (canonical). If not provided,
        computed via autograd.
    energy_weight:
        Weight for the energy term (0 disables it).
    return_parts:
        If True, returns a dict with individual components as well.

    Returns
    -------
    torch.Tensor  (or Dict[str, torch.Tensor] if return_parts=True)
        Total physics loss (and optionally its components).
    """
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"physics_loss expects t shaped (N,1); got {tuple(t.shape)}")
    if ic.ndim != 2 or ic.shape[1] != 6:
        raise ValueError(f"physics_loss expects ic shaped (N,6); got {tuple(ic.shape)}")

    # Ensure t supports autograd (leaf preferred)
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)

    state_nd = state_nd if state_nd is not None else model(t, ic)

    r_nd = state_nd[:, 0:3]
    v_nd = state_nd[:, 3:6]

    drdt_nd = drdt_nd if drdt_nd is not None else time_derivative(r_nd, t)
    dvdt_nd = dvdt_nd if dvdt_nd is not None else time_derivative(v_nd, t)

    a_nd = accel_j2_canonical_from_state(state_nd, j2=float(model.j2) if hasattr(model, "j2") else float(model.J2))

    loss_kin = torch.mean((drdt_nd - v_nd) ** 2)
    loss_dyn = torch.mean((dvdt_nd - a_nd) ** 2)
    loss_total = loss_kin + loss_dyn

    loss_energy = None
    if energy_weight and float(energy_weight) != 0.0:
        # H(t)
        H_t = hamiltonian_canonical_torch(state_nd, j2=float(model.j2) if hasattr(model, "j2") else float(model.J2))  # (N,1)

        # H0 from IC
        r0 = ic[:, 0:3]
        v0 = ic[:, 3:6]
        H_0 = 0.5 * torch.sum(v0 * v0, dim=1, keepdim=True) + potential_j2_canonical_torch(
            r0,
            j2=float(model.j2) if hasattr(model, "j2") else float(model.J2),
        )

        loss_energy = torch.mean((H_t - H_0) ** 2)
        loss_total = loss_total + float(energy_weight) * loss_energy

    if not return_parts:
        return loss_total

    parts: Dict[str, "torch.Tensor"] = {
        "loss_total": loss_total,
        "loss_kin": loss_kin,
        "loss_dyn": loss_dyn,
    }
    if loss_energy is not None:
        parts["loss_energy"] = loss_energy
    return parts


# ============================================================================
# 9) Checkpoint helpers
# ============================================================================

def save_checkpoint(
    path: str,
    model: OrbitPINN,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint.

    The checkpoint format is:
      {
        "state_dict": model.state_dict(),
        "extra": {...}   # optional metadata (config, scaler, metrics, etc.)
      }
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpointing.")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"state_dict": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: OrbitPINN,
    *,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a model checkpoint and restore the model parameters.

    Parameters
    ----------
    path:
        Checkpoint file path.
    model:
        Instantiated model with matching architecture.
    map_location:
        Device mapping passed to torch.load (e.g., "cpu", "cuda").
    strict:
        Passed to model.load_state_dict. If False, allows partial loading.

    Returns
    -------
    Dict[str, Any]
        The "extra" metadata dictionary (may be empty).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for checkpointing.")

    payload = torch.load(path, map_location=map_location)
    if "state_dict" not in payload:
        raise KeyError("Checkpoint is missing required key: 'state_dict'")

    model.load_state_dict(payload["state_dict"], strict=strict)
    return payload.get("extra", {}) or {}


# ============================================================================
# 10) Metrics
# ============================================================================

def error_metrics(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Compute position/velocity error statistics in physical units.

    Parameters
    ----------
    truth, pred:
        Arrays shaped (N,6) representing [x,y,z,vx,vy,vz] with:
          - position: km
          - velocity: km/s

    Returns
    -------
    Dict[str, float]
        Summary stats for:
          - position error [km]
          - velocity error [m/s]
        using mean / max / RMSE / p95.
    """
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 6)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 6)

    r_true_km, v_true_km_s = truth[:, :3], truth[:, 3:]
    r_pred_km, v_pred_km_s = pred[:, :3], pred[:, 3:]

    pos_err_km = np.linalg.norm(r_true_km - r_pred_km, axis=1)
    vel_err_m_s = np.linalg.norm(v_true_km_s - v_pred_km_s, axis=1) * 1000.0  # km/s -> m/s

    def summarize(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64).ravel()
        rmse = float(np.sqrt(np.mean(x * x))) if x.size else float("nan")
        return {
            "mean": float(np.mean(x)) if x.size else float("nan"),
            "max": float(np.max(x)) if x.size else float("nan"),
            "rmse": rmse,
            "p95": float(np.percentile(x, 95)) if x.size else float("nan"),
        }

    pos = summarize(pos_err_km)
    vel = summarize(vel_err_m_s)

    return {
        "pos_mean_km": pos["mean"],
        "pos_max_km": pos["max"],
        "pos_rmse_km": pos["rmse"],
        "pos_p95_km": pos["p95"],
        "vel_mean_m_s": vel["mean"],
        "vel_max_m_s": vel["max"],
        "vel_rmse_m_s": vel["rmse"],
        "vel_p95_m_s": vel["p95"],
    }
