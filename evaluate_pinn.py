# -*- coding: utf-8 -*-
"""
evaluate_pinn.py
================
Kaydedilen OrbitPINN checkpoint'ini yükler, verilen IC ile:
- PINN tahmini üretir
- aynı IC ile solve_ivp ground-truth üretir
- metrikler + görseller + metrics.json üretir (opsiyonel)

UI kullanımı (önerilen)
-----------------------
    from evaluate_pinn import EvalConfig, evaluate
    cfg = EvalConfig(checkpoint="checkpoints/orbit_pinn.pt", out_dir="eval_outputs")
    report = evaluate(cfg)
"""
# =======================================================================
# 0.                            IMPORTS
# =======================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import torch

from orbit_core import (
    MU_EARTH, R_EARTH,
    seed_everything,
    get_device,
    CanonicalScaler, ScalerConfig,
    OrbitPINN,
    satellite_dynamics,
    coe_to_rv,
    error_metrics,
)


# ============================================================================
# 1) Config
# ============================================================================

@dataclass(slots=True)
class EvalConfig:
    checkpoint: str = "checkpoints/orbit_pinn.pt"
    out_dir: str = "eval_outputs"

    # Zaman grid
    duration_s: float = 86400.0
    dt_s: float = 60.0
    include_end: bool = True

    # Truth integrasyon
    method: str = "DOP853"
    rtol: float = 1e-12
    atol: float = 1e-12

    seed: int = 123

    # IC tanımı
    # 1) direkt state (km, km/s)
    y0: Optional[Tuple[float, float, float, float, float, float]] = None

    # 2) COE-ish
    use_coe: bool = True
    perigee_alt_km: float = 700.0
    e: float = 0.01
    inc_deg: float = 45.0
    raan_deg: float = 10.0
    argp_deg: float = 20.0
    nu_deg: float = 0.0

    # Outputs
    save_plots: bool = True
    save_json: bool = True
    return_arrays: bool = False   # True -> truth/pred/t gibi büyük array'leri de döndürür


# ============================================================================
# 2) Utilities
# ============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_t_eval(duration_s: float, dt_s: float, include_end: bool = True) -> np.ndarray:
    """t_eval'i her zaman [0, duration] içinde ve monotonik tut."""
    if duration_s <= 0:
        raise ValueError(f"duration_s > 0 olmalı (verilen={duration_s}).")
    if dt_s <= 0:
        raise ValueError(f"dt_s > 0 olmalı (verilen={dt_s}).")

    t = np.arange(0.0, duration_s, dt_s, dtype=np.float64)

    if include_end:
        if t.size == 0 or t[-1] < duration_s:
            t = np.append(t, np.float64(duration_s))
        else:
            t[-1] = np.float64(duration_s)

    if t.size == 0:
        t = np.array([0.0, duration_s], dtype=np.float64) if include_end else np.array([0.0], dtype=np.float64)

    t = np.clip(t, 0.0, duration_s)
    t = np.unique(t)  # monotonik + duplicate temizliği
    if t.size < 2 and include_end:
        t = np.array([0.0, float(duration_s)], dtype=np.float64)
    return t.astype(np.float64)


def _get_initial_state(cfg: EvalConfig) -> np.ndarray:
    """IC'yi (km, km/s) olarak döndür."""
    if cfg.y0 is not None:
        y0 = np.array(cfg.y0, dtype=np.float64).reshape(-1)
        if y0.shape != (6,):
            raise ValueError("y0 6 elemanlı olmalı: (x,y,z,vx,vy,vz)")
        if not np.all(np.isfinite(y0)):
            raise ValueError("y0 içinde NaN/Inf var.")
        return y0

    if not cfg.use_coe:
        raise ValueError("Ne y0 verildi ne de use_coe=True. En az biri gerekli.")

    inc = np.deg2rad(float(cfg.inc_deg))
    raan = np.deg2rad(float(cfg.raan_deg))
    argp = np.deg2rad(float(cfg.argp_deg))
    nu = np.deg2rad(float(cfg.nu_deg))

    r_p = float(R_EARTH) + float(cfg.perigee_alt_km)
    if r_p <= 0.0:
        raise ValueError(f"perigee radius <= 0 çıktı: r_p={r_p}")

    e = float(cfg.e)
    if not (0.0 <= e < 1.0):
        raise ValueError(f"e için 0 <= e < 1 beklenir (verilen={e}).")

    a = r_p / (1.0 - e)
    y0 = coe_to_rv(a=a, e=e, inc=inc, raan=raan, argp=argp, nu=nu, mu=float(MU_EARTH)).astype(np.float64)

    if y0.shape != (6,) or (not np.all(np.isfinite(y0))):
        raise RuntimeError("coe_to_rv geçersiz state üretti.")
    return y0


def _jsonable(x: Any) -> Any:
    """json.dump için dayanıklı dönüştürücü (büyük array'leri şişirmez)."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        if x.size <= 128:
            return x.tolist()
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if torch.is_tensor(x):
        return {"__tensor__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    return str(x)


def _load_checkpoint_raw(path: str, map_location: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """state_dict ve extra/meta/config bilgilerini toleranslı şekilde yükle."""
    blob = torch.load(path, map_location=map_location)

    if not isinstance(blob, dict):
        raise ValueError("Checkpoint formatı desteklenmiyor (dict değil).")

    # state dict anahtar toleransı
    state: Dict[str, Any] = {}
    for key in ("state_dict", "model_state_dict", "model_state", "net_state_dict", "pinn_state_dict"):
        if key in blob and isinstance(blob[key], dict):
            state = blob[key]
            break

    # bazı durumlarda state doğrudan blob'un kendisi olabilir
    if not state and any(torch.is_tensor(v) for v in blob.values()):
        state = blob  # type: ignore

    if not state:
        raise ValueError("Checkpoint içinde model state_dict bulunamadı.")

    extra: Dict[str, Any] = {}
    for key in ("extra", "meta", "config"):
        if key in blob and isinstance(blob[key], dict):
            extra = blob[key]
            break

    # extra yoksa: heuristik
    if not extra:
        extra = {k: v for k, v in blob.items() if k not in state and not torch.is_tensor(v)}

    return state, extra


def _strip_prefix(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not prefix:
        return state
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _filter_init_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """OrbitPINN init arglarını filtrele (fazla anahtarlar patlatmasın)."""
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid}


def _build_model_from_extra(extra: Dict[str, Any]) -> OrbitPINN:
    model_cfg = extra.get("model_config", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    # fallback (eski ckpt)
    model_cfg.setdefault("hidden", int(model_cfg.get("hidden", 128)))
    model_cfg.setdefault("depth", int(model_cfg.get("depth", 3)))

    kwargs = _filter_init_kwargs(OrbitPINN, model_cfg)
    return OrbitPINN(**kwargs)


def _infer_dtype_from_extra(extra: Dict[str, Any]) -> torch.dtype:
    train_cfg = extra.get("train_config", {})
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    s = str(train_cfg.get("dtype", "float32")).lower().strip()
    if s in ("float64", "fp64", "double"):
        return torch.float64
    return torch.float32


def _build_scaler_from_extra(extra: Dict[str, Any]) -> CanonicalScaler:
    scaler_cfg = ScalerConfig()
    scaler_blob = extra.get("scaler", {})
    if isinstance(scaler_blob, dict):
        cfg_blob = scaler_blob.get("cfg", scaler_blob.get("config", {}))
        if isinstance(cfg_blob, dict):
            try:
                scaler_cfg = ScalerConfig(
                    r_ref_km=float(cfg_blob.get("r_ref_km", float(R_EARTH))),
                    mu_km3_s2=float(cfg_blob.get("mu_km3_s2", float(MU_EARTH))),
                )
            except Exception:
                scaler_cfg = ScalerConfig()
    return CanonicalScaler(scaler_cfg)


def _load_model_and_extra(cfg: EvalConfig, device: torch.device) -> Tuple[OrbitPINN, CanonicalScaler, Dict[str, Any]]:
    state, extra = _load_checkpoint_raw(cfg.checkpoint, map_location=str(device))

    # dtype (training ile uyum)
    dtype = _infer_dtype_from_extra(extra)
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    model = _build_model_from_extra(extra).to(device)

    # state_dict prefix toleransı
    prefixes = ("", "model.", "net.", "pinn.", "module.")
    loaded = False

    for pref in prefixes:
        try:
            sd = _strip_prefix(state, pref) if pref else state
            model.load_state_dict(sd, strict=True)
            loaded = True
            break
        except Exception:
            continue

    if not loaded:
        # strict=False last resort
        for pref in prefixes:
            try:
                sd = _strip_prefix(state, pref) if pref else state
                model.load_state_dict(sd, strict=False)
                loaded = True
                break
            except Exception:
                continue

    if not loaded:
        raise RuntimeError("Model state_dict yüklenemedi (prefix/strict denemeleri başarısız).")

    model.eval()

    scaler = _build_scaler_from_extra(extra)
    # dtype'ı evaluate boyunca kullanacağız; dışarı sızmasın diye evaluate() finally'de geri alacağız
    extra["_eval_internal_prev_default_dtype"] = prev_default  # internal marker
    return model, scaler, extra


def _solve_truth(
    y0: np.ndarray,
    t_eval: np.ndarray,
    *,
    cfg: EvalConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        fun=satellite_dynamics,
        t_span=(0.0, float(cfg.duration_s)),
        y0=y0.astype(np.float64),
        t_eval=t_eval.astype(np.float64),
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        method=str(cfg.method),
    )
    if not sol.success or sol.t is None or sol.y is None:
        raise RuntimeError(f"Ground-truth solve_ivp başarısız: {getattr(sol, 'message', '')}")
    t = sol.t.reshape(-1, 1).astype(np.float64)
    y = sol.y.T.astype(np.float64)  # (N,6)
    return t, y


def _predict_pinn(
    model: OrbitPINN,
    scaler: CanonicalScaler,
    *,
    device: torch.device,
    t_truth: np.ndarray,
    y0: np.ndarray,
) -> np.ndarray:
    # IC replicate
    ic_raw = np.tile(y0.reshape(1, 6), (t_truth.shape[0], 1)).astype(np.float64)

    # canonical
    t_n, ic_n = scaler.transform(t_truth, ic_raw)

    dtype = torch.get_default_dtype()
    t_tensor = torch.tensor(t_n, dtype=dtype, device=device)
    ic_tensor = torch.tensor(ic_n, dtype=dtype, device=device)

    with torch.no_grad():
        pred_n = model(t_tensor, ic_tensor).detach().cpu().numpy()

    _, y_pred = scaler.inverse_transform(t_n, pred_n)
    return y_pred.astype(np.float64)


def _save_plots(
    out_dir: str,
    *,
    t_s: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, str]:
    _ensure_dir(out_dir)

    hours = (t_s[:, 0] / 3600.0).astype(np.float64)
    r_t = truth[:, :3]
    v_t = truth[:, 3:]
    r_p = pred[:, :3]
    v_p = pred[:, 3:]

    pos_err = np.linalg.norm(r_t - r_p, axis=1)
    vel_err = np.linalg.norm(v_t - v_p, axis=1) * 1000.0  # m/s

    paths: Dict[str, str] = {}

    # --- 3D orbit ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    ax.plot_surface(
        float(R_EARTH) * np.cos(u) * np.sin(v),
        float(R_EARTH) * np.sin(u) * np.sin(v),
        float(R_EARTH) * np.cos(v),
        alpha=0.12
    )

    ax.plot(r_t[:, 0], r_t[:, 1], r_t[:, 2], label="Truth (solve_ivp)")
    ax.plot(r_p[:, 0], r_p[:, 1], r_p[:, 2], label="PINN (pred)", linestyle="--")

    ax.set_title("3D Yörünge Karşılaştırması")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()
    plt.tight_layout()

    p3d = os.path.join(out_dir, "orbit_3d_compare.png")
    plt.savefig(p3d, dpi=160)
    plt.close(fig)
    paths["orbit_3d"] = p3d

    # --- components (position) ---
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hours, r_t[:, 0], label="Truth x", alpha=0.6)
    plt.plot(hours, r_p[:, 0], label="Pred x", linestyle="--")
    plt.plot(hours, r_t[:, 1], label="Truth y", alpha=0.6)
    plt.plot(hours, r_p[:, 1], label="Pred y", linestyle="--")
    plt.plot(hours, r_t[:, 2], label="Truth z", alpha=0.6)
    plt.plot(hours, r_p[:, 2], label="Pred z", linestyle="--")
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Konum [km]")
    plt.title("Konum Bileşenleri")
    plt.grid(alpha=0.3)
    plt.legend(ncols=2)
    plt.tight_layout()

    pcomp = os.path.join(out_dir, "pos_components.png")
    plt.savefig(pcomp, dpi=160)
    plt.close(fig)
    paths["pos_components"] = pcomp

    # --- position error ---
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hours, pos_err, label="Konum hatası [km]")
    plt.yscale("log")
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Hata")
    plt.title("Konum Hatası (log)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    perror = os.path.join(out_dir, "pos_error.png")
    plt.savefig(perror, dpi=160)
    plt.close(fig)
    paths["pos_error"] = perror

    # --- velocity error ---
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hours, vel_err, label="Hız hatası [m/s]")
    plt.yscale("log")
    plt.xlabel("Zaman [saat]")
    plt.ylabel("Hata")
    plt.title("Hız Hatası (log)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pverr = os.path.join(out_dir, "vel_error.png")
    plt.savefig(pverr, dpi=160)
    plt.close(fig)
    paths["vel_error"] = pverr

    return paths


# ============================================================================
# 3) Public API
# ============================================================================

def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    """
    Evaluate OrbitPINN vs solve_ivp truth for a single initial condition.

    Returns a report dict. If cfg.return_arrays=True, report includes large arrays.
    """
    seed_everything(cfg.seed)
    device = get_device()

    _ensure_dir(cfg.out_dir)

    model: OrbitPINN
    scaler: CanonicalScaler
    extra: Dict[str, Any]

    # default dtype changes are global; keep them contained
    prev_default_dtype = torch.get_default_dtype()

    try:
        model, scaler, extra = _load_model_and_extra(cfg, device)

        # _load_model_and_extra içine internal marker koyduk
        prev_default_dtype = extra.pop("_eval_internal_prev_default_dtype", prev_default_dtype)

        # IC
        y0 = _get_initial_state(cfg)

        # Truth
        t_eval = _make_t_eval(cfg.duration_s, cfg.dt_s, include_end=cfg.include_end)
        t_truth, y_truth = _solve_truth(y0, t_eval, cfg=cfg)

        # PINN predict
        y_pred = _predict_pinn(model, scaler, device=device, t_truth=t_truth, y0=y0)

        # Metrics
        metrics = error_metrics(y_truth, y_pred)

        # Outputs
        plots: Dict[str, str] = {}
        if cfg.save_plots:
            plots = _save_plots(cfg.out_dir, t_s=t_truth, truth=y_truth, pred=y_pred)

        report: Dict[str, Any] = {
            "eval_config": asdict(cfg),
            "checkpoint": cfg.checkpoint,
            "y0_km_km_s": y0.tolist(),
            "metrics": _jsonable(metrics),
            "plots": plots,
            "checkpoint_extra": _jsonable(extra),
        }

        if cfg.return_arrays:
            report["t_truth_s"] = t_truth
            report["truth_state_km_km_s"] = y_truth
            report["pred_state_km_km_s"] = y_pred

        if cfg.save_json:
            metrics_path = os.path.join(cfg.out_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=_jsonable)
            report["metrics_json"] = metrics_path

        # Console summary (UI’da log paneli varsa faydalı)
        print("[OK] Evaluation completed.")
        for k, v in metrics.items():
            try:
                print(f"  {k}: {float(v):.6g}")
            except Exception:
                print(f"  {k}: {v}")
        if cfg.save_json:
            print(f"[OK] metrics.json: {report.get('metrics_json')}")
        return report

    finally:
        torch.set_default_dtype(prev_default_dtype)

