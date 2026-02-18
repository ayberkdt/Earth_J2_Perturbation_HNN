# -*- coding: utf-8 -*-
"""
train_pinn.py
=============
CSV dataset'i okuyup OrbitPINN modelini eğitir ve checkpoint/meta/plot çıktıları üretir.

UI kullanımı (önerilen)
-----------------------
    from train_pinn import TrainConfig, train
    cfg = TrainConfig(dataset_csv="dataset/orbit_j2_dataset_v2.csv", epochs=2000)
    ckpt_path = train(cfg)

Model
-----
input  = [t, x0, y0, z0, vx0, vy0, vz0]  (canonical)
output = [x, y, z, vx, vy, vz]           (canonical)

Loss
----
L = W_DATA * (MSE(y_pred, y_true) + alpha_v * MSE(d r_pred/dt, v_true))
  + W_PHYS * physics_loss(model, t, ic, ...)
"""

# =======================================================================
# 0.                            IMPORTS
# =======================================================================

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ----------------------------------------------------------------------
# orbit_core imports (backward-compatible)
# ----------------------------------------------------------------------
from orbit_core import (
    seed_everything,
    get_device,
    CanonicalScaler, ScalerConfig,
    OrbitDataset,
    OrbitPINN,
    save_checkpoint,
    load_checkpoint,
)

# d_dt -> time_derivative refactor uyumu
try:
    from orbit_core import time_derivative as d_dt  # new name
except Exception:
    from orbit_core import d_dt  # old name

# physics_loss imza refactor uyumu
from orbit_core import physics_loss


# ============================================================================
# 1).                               Config
# ============================================================================

@dataclass(slots=True)
class TrainConfig:
    # -------------------------
    # Data
    # -------------------------
    dataset_csv: str = "dataset/orbit_j2_dataset_v2.csv"
    batch_size: int = 2048
    num_workers: int = 0
    val_split: float = 0.02          # 0 -> kapalı
    val_batches: int = 2             # val için kaç batch ölçelim (hız için)

    # -------------------------
    # Training
    # -------------------------
    epochs: int = 2000
    lr: float = 1e-3
    seed: int = 42
    dtype: str = "float32"           # "float32" | "float64"
    grad_clip_norm: float = 0.0      # 0 -> kapalı

    # -------------------------
    # Loss weights
    # -------------------------
    w_data: float = 1.0
    w_phys: float = 0.1
    phys_warmup_epochs: int = 0      # 0 -> sabit, >0 -> lineer ramp

    # Derivative data loss: ||dr/dt - v_data||^2
    use_derivative_data_loss: bool = True
    alpha_v: float = 1.0

    # Energy loss inside physics_loss
    use_energy_loss: bool = True
    energy_weight: float = 1.0

    # -------------------------
    # Adaptive weighting (GradNorm-lite)
    # -------------------------
    adaptive_weighting: bool = True
    aw_update_every: int = 1
    aw_beta: float = 0.5
    aw_ema: float = 0.9
    aw_min: float = 1e-4
    aw_max: float = 1e4

    # -------------------------
    # Curriculum learning (time-window)
    # -------------------------
    curriculum: bool = True
    curr_e1: int = 500
    curr_e2: int = 1000
    curr_t1_s: float = 3600.0        # 1 saat
    curr_t2_s: float = 21600.0       # 6 saat
    curr_t3_s: float = 86400.0       # 24 saat

    # -------------------------
    # Optimizer schedule: Adam -> (optional) L-BFGS
    # -------------------------
    use_lbfgs: bool = True
    lbfgs_fraction: float = 0.2
    lbfgs_max_iter: int = 200
    lbfgs_history_size: int = 50
    lbfgs_lr: float = 1.0
    lbfgs_line_search: str = "strong_wolfe"
    lbfgs_sample_size: int = 8192

    # -------------------------
    # Model config
    # -------------------------
    hidden: int = 128
    depth: int = 3
    arch: str = "deeponet_phase"     # mlp | fourier_mlp | ... | deeponet_phase

    # Fourier (t)
    fourier_features: int = 16
    fourier_min_freq: float = 0.01
    fourier_max_freq: float = 10.0
    fourier_include_input: bool = True
    fourier_log_sampling: bool = True

    # Phase
    phase_harmonics: int = 8
    phase_include_t: bool = True
    phase_include_phase: bool = False

    # DeepONet
    deeponet_latent: int = 64
    deeponet_branch_aug: bool = True

    # SIREN
    siren_w0_initial: float = 30.0
    siren_w0: float = 1.0

    # Hard constraint in model
    hard_constraint: bool = True

    # -------------------------
    # I/O
    # -------------------------
    out_dir: str = "checkpoints"
    ckpt_name: str = "orbit_pinn.pt"
    save_every: int = 200            # epoch periyodu, 0 -> kapalı
    log_every: int = 10

    # Artifacts
    save_plot: bool = True
    save_meta: bool = True

    # Resume
    resume_from: Optional[str] = None



# ============================================================================
# 2.                                 Helpers
# ============================================================================

def _dtype_from_str(s: str) -> torch.dtype:
    s = str(s).lower().strip()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported dtype: {s}")


def _build_model_kwargs(cfg: TrainConfig) -> Dict[str, Any]:
    """OrbitPINN init kwargs (tek yerde toplu dursun)."""
    return {
        "hidden": cfg.hidden,
        "depth": cfg.depth,
        "arch": cfg.arch,
        "fourier_features": cfg.fourier_features,
        "fourier_max_freq": cfg.fourier_max_freq,
        "fourier_min_freq": cfg.fourier_min_freq,
        "fourier_include_input": cfg.fourier_include_input,
        "fourier_log_sampling": cfg.fourier_log_sampling,
        "phase_harmonics": cfg.phase_harmonics,
        "phase_include_t": cfg.phase_include_t,
        "phase_include_phase": cfg.phase_include_phase,
        "deeponet_latent": cfg.deeponet_latent,
        "deeponet_branch_aug": cfg.deeponet_branch_aug,
        "siren_w0_initial": cfg.siren_w0_initial,
        "siren_w0": cfg.siren_w0,
        "hard_constraint": cfg.hard_constraint,
    }


def _split_train_val(n: int, val_split: float, seed: int) -> Tuple[List[int], Optional[List[int]]]:
    if (not val_split) or float(val_split) <= 0.0:
        return list(range(n)), None
    val_n = int(round(n * float(val_split)))
    val_n = max(1, min(val_n, n - 1))  # en az 1 val, en az 1 train
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    return train_idx, val_idx


def _phys_warmup_factor(cfg: TrainConfig, epoch: int) -> float:
    if cfg.phys_warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(cfg.phys_warmup_epochs))


def _curriculum_tmax_canonical(cfg: TrainConfig, epoch: int, *, dataset: OrbitDataset, scaler: CanonicalScaler) -> float:
    """Return t_max in canonical units for current epoch."""
    t_max_dataset = float(dataset.t.max().item())

    if not cfg.curriculum:
        return t_max_dataset

    if epoch < int(cfg.curr_e1):
        tmax_s = float(cfg.curr_t1_s)
    elif epoch < int(cfg.curr_e2):
        tmax_s = float(cfg.curr_t2_s)
    else:
        tmax_s = float(cfg.curr_t3_s)

    tmax_n = tmax_s / float(scaler.TU)
    return min(tmax_n, t_max_dataset)


def _masked_batch(
    t: torch.Tensor, ic: torch.Tensor, y: Optional[torch.Tensor], *, tmax_n: float, enabled: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not enabled:
        return t, ic, y
    msk = (t[:, 0] <= float(tmax_n))
    if not bool(msk.any()):
        return t[:0], ic[:0], (y[:0] if y is not None else None)
    t2 = t[msk]
    ic2 = ic[msk]
    y2 = (y[msk] if y is not None else None)
    return t2, ic2, y2


def _grad_norm(model: nn.Module, loss: torch.Tensor, device: torch.device) -> torch.Tensor:
    """||∇_θ loss||_2 (detached)."""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    sq = None
    for g in grads:
        if g is None:
            continue
        v = (g.detach() ** 2).sum()
        sq = v if sq is None else (sq + v)
    if sq is None:
        return torch.tensor(0.0, device=device)
    return torch.sqrt(sq + 1e-12)


def _call_physics_loss(
    model: OrbitPINN,
    t: torch.Tensor,
    ic: torch.Tensor,
    *,
    y_pred: torch.Tensor,
    drdt_pred: torch.Tensor,
    energy_weight: float,
) -> torch.Tensor:
    """
    orbit_core.physics_loss imzası değişmiş olsa bile çalışabilsin diye çağrı sarmalayıcı.
    - Eski: physics_loss(model, t, ic, state=..., drdt=..., energy_weight=...)
    - Yeni: physics_loss(model, t, ic, state_nd=..., drdt_nd=..., energy_weight=...)
    """
    try:
        return physics_loss(
            model,
            t,
            ic,
            state=y_pred,
            drdt=drdt_pred,
            energy_weight=float(energy_weight),
        )
    except TypeError:
        return physics_loss(
            model,
            t,
            ic,
            state_nd=y_pred,
            drdt_nd=drdt_pred,
            energy_weight=float(energy_weight),
        )


@torch.no_grad()
def _eval_data_terms(
    model: OrbitPINN,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    tmax_n: float,
) -> Tuple[float, float, int]:
    """
    Evaluate data MSE (+ optional derivative data loss) over at most cfg.val_batches batches.
    Returns: (mean_data_mse, mean_dv_mse, n_batches_used)
    """
    model.eval()
    s_data = 0.0
    s_dv = 0.0
    n_batches = 0

    for bt, bic, by in loader:
        bt = bt.to(device, non_blocking=True)
        bic = bic.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        bt, bic, by = _masked_batch(bt, bic, by, tmax_n=tmax_n, enabled=cfg.curriculum)
        if bt.numel() == 0:
            continue

        # only require grad if derivative data loss is enabled
        need_grad = bool(cfg.use_derivative_data_loss and float(cfg.alpha_v) != 0.0)
        bt = bt.clone().detach().requires_grad_(need_grad)

        yp = model(bt, bic)
        ld = loss_fn(yp, by)
        s_data += float(ld.item())

        if need_grad:
            drdt_v = d_dt(yp[:, 0:3], bt)
            dv = torch.mean((drdt_v - by[:, 3:6]) ** 2)
            s_dv += float(dv.item())

        n_batches += 1
        if n_batches >= max(1, int(cfg.val_batches)):
            break

    if n_batches == 0:
        return float("nan"), float("nan"), 0
    return s_data / n_batches, s_dv / n_batches, n_batches


def _eval_phys_term(
    model: OrbitPINN,
    loader: DataLoader,
    *,
    device: torch.device,
    cfg: TrainConfig,
    tmax_n: float,
    energy_weight: float,
) -> Tuple[float, int]:
    """Evaluate physics loss (needs grad wrt t) over at most cfg.val_batches batches."""
    model.eval()
    s_phys = 0.0
    n_batches = 0

    for bt, bic, _ in loader:
        bt = bt.to(device, non_blocking=True)
        bic = bic.to(device, non_blocking=True)

        bt, bic, _ = _masked_batch(bt, bic, None, tmax_n=tmax_n, enabled=cfg.curriculum)
        if bt.numel() == 0:
            continue

        bt = bt.clone().detach().requires_grad_(True)

        yp = model(bt, bic)
        drdt_v = d_dt(yp[:, 0:3], bt)

        lp = _call_physics_loss(
            model,
            bt,
            bic,
            y_pred=yp,
            drdt_pred=drdt_v,
            energy_weight=float(energy_weight),
        )
        s_phys += float(lp.item())
        n_batches += 1
        if n_batches >= max(1, int(cfg.val_batches)):
            break

    if n_batches == 0:
        return float("nan"), 0
    return s_phys / n_batches, n_batches



# ============================================================================
# 3.                                 Train
# ============================================================================

def train(cfg: TrainConfig) -> str:
    """
    Train OrbitPINN model according to cfg and save outputs under cfg.out_dir.

    Returns
    -------
    str
        Final checkpoint path.
    """
    seed_everything(cfg.seed)
    device = get_device()

    dtype = _dtype_from_str(cfg.dtype)
    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    try:
        scaler = CanonicalScaler(ScalerConfig())
        dataset = OrbitDataset(cfg.dataset_csv, scaler, dtype=dtype)

        train_idx, val_idx = _split_train_val(len(dataset), cfg.val_split, cfg.seed)
        train_ds = Subset(dataset, train_idx) if val_idx is not None else dataset
        val_ds = Subset(dataset, val_idx) if val_idx is not None else None

        pin_memory = bool(device.type == "cuda")
        loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            drop_last=False,
            num_workers=int(cfg.num_workers),
            pin_memory=pin_memory,
        )

        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=pin_memory,
            )

        model = OrbitPINN(**_build_model_kwargs(cfg)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
        loss_fn = nn.MSELoss()

        history: Dict[str, List[float]] = {
            "train_total": [],
            "train_data": [],
            "train_data_dv": [],
            "train_phys": [],
            "val_total": [],
            "val_data": [],
            "val_data_dv": [],
            "val_phys": [],
            "w_data": [],
            "w_phys": [],
        }

        # dynamic weights (adaptive weighting updates these)
        w_data_dyn = float(cfg.w_data)
        w_phys_dyn = float(cfg.w_phys)

        # Adam epochs count if L-BFGS enabled
        adam_epochs = int(cfg.epochs)
        if cfg.use_lbfgs and float(cfg.lbfgs_fraction) > 0.0:
            adam_epochs = max(1, int(round(float(cfg.epochs) * (1.0 - float(cfg.lbfgs_fraction)))))

        os.makedirs(cfg.out_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)
        best_path = os.path.join(cfg.out_dir, "best_" + cfg.ckpt_name)

        # optional resume
        if cfg.resume_from:
            extra = load_checkpoint(cfg.resume_from, model, map_location=str(device))
            print(f"[INFO] Resume: {cfg.resume_from} | extra_keys={list(extra.keys())}", flush=True)

        best_val = float("inf")
        t0 = time.time()
        print(
            f"[INFO] Training start | device={device} | dtype={dtype} | "
            f"train_batches/epoch={len(loader)} | val={'on' if val_loader else 'off'}",
            flush=True,
        )

        # -----------------------
        # 1) Adam stage
        # -----------------------
        for epoch in range(adam_epochs):
            model.train()

            warm = _phys_warmup_factor(cfg, epoch)
            tmax_n = _curriculum_tmax_canonical(cfg, epoch, dataset=dataset, scaler=scaler)

            # epoch weights (may be updated once by adaptive weighting)
            w_data_now = float(w_data_dyn)
            w_phys_now = float(w_phys_dyn) * float(warm)

            sum_total = 0.0
            sum_data = 0.0
            sum_data_dv = 0.0
            sum_phys = 0.0
            n_steps = 0

            did_aw = False

            for t, ic, y_true in loader:
                t = t.to(device, non_blocking=True)
                ic = ic.to(device, non_blocking=True)
                y_true = y_true.to(device, non_blocking=True)

                # Curriculum mask
                t, ic, y_true = _masked_batch(t, ic, y_true, tmax_n=tmax_n, enabled=cfg.curriculum)
                if t.numel() == 0:
                    continue

                # Leaf t for autograd
                t = t.clone().detach().requires_grad_(True)

                optimizer.zero_grad(set_to_none=True)

                y_pred = model(t, ic)
                loss_data_mse = loss_fn(y_pred, y_true)

                # dr/dt prediction (reused)
                drdt_pred = d_dt(y_pred[:, 0:3], t)

                loss_data_dv = torch.tensor(0.0, device=device)
                if cfg.use_derivative_data_loss and float(cfg.alpha_v) != 0.0:
                    v_true = y_true[:, 3:6]
                    loss_data_dv = torch.mean((drdt_pred - v_true) ** 2)

                loss_data_total = loss_data_mse + float(cfg.alpha_v) * loss_data_dv

                e_w = float(cfg.energy_weight) if cfg.use_energy_loss else 0.0
                loss_phys = _call_physics_loss(
                    model,
                    t,
                    ic,
                    y_pred=y_pred,
                    drdt_pred=drdt_pred,
                    energy_weight=e_w,
                )

                # ---- Adaptive weighting (once per epoch, first usable batch) ----
                if (
                    cfg.adaptive_weighting
                    and (not did_aw)
                    and (epoch % max(1, int(cfg.aw_update_every)) == 0)
                ):
                    gd = float(_grad_norm(model, loss_data_total, device).item())
                    gp = float(_grad_norm(model, loss_phys, device).item())
                    eps = 1e-12
                    if gd > eps and gp > eps:
                        target = 0.5 * (gd + gp)
                        beta = float(cfg.aw_beta)

                        w_data_new = w_data_dyn * (target / (gd + eps)) ** beta
                        w_phys_new = w_phys_dyn * (target / (gp + eps)) ** beta

                        ema = float(cfg.aw_ema)
                        w_data_dyn = ema * w_data_dyn + (1.0 - ema) * w_data_new
                        w_phys_dyn = ema * w_phys_dyn + (1.0 - ema) * w_phys_new

                        # normalize to keep sum stable
                        s0 = float(cfg.w_data) + float(cfg.w_phys)
                        s1 = w_data_dyn + w_phys_dyn + eps
                        scale = s0 / s1
                        w_data_dyn *= scale
                        w_phys_dyn *= scale

                        # clamp
                        w_data_dyn = float(min(max(w_data_dyn, float(cfg.aw_min)), float(cfg.aw_max)))
                        w_phys_dyn = float(min(max(w_phys_dyn, float(cfg.aw_min)), float(cfg.aw_max)))

                        w_data_now = float(w_data_dyn)
                        w_phys_now = float(w_phys_dyn) * float(warm)
                        did_aw = True

                loss = w_data_now * loss_data_total + w_phys_now * loss_phys
                loss.backward()

                if cfg.grad_clip_norm and float(cfg.grad_clip_norm) > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))

                optimizer.step()

                sum_total += float(loss.item())
                sum_data += float(loss_data_mse.item())
                sum_data_dv += float(loss_data_dv.item())
                sum_phys += float(loss_phys.item())
                n_steps += 1

            denom = max(1, n_steps)
            train_total = sum_total / denom
            train_data = sum_data / denom
            train_data_dv = sum_data_dv / denom
            train_phys = sum_phys / denom

            history["train_total"].append(train_total)
            history["train_data"].append(train_data)
            history["train_data_dv"].append(train_data_dv)
            history["train_phys"].append(train_phys)
            history["w_data"].append(float(w_data_dyn))
            history["w_phys"].append(float(w_phys_dyn))

            # ---- Validation (optional) ----
            val_total = float("nan")
            val_data = float("nan")
            val_data_dv = float("nan")
            val_phys = float("nan")

            if val_loader is not None:
                val_data, val_data_dv, _ = _eval_data_terms(
                    model,
                    val_loader,
                    device=device,
                    loss_fn=loss_fn,
                    cfg=cfg,
                    tmax_n=tmax_n,
                )

                e_w = float(cfg.energy_weight) if cfg.use_energy_loss else 0.0
                val_phys, _ = _eval_phys_term(
                    model,
                    val_loader,
                    device=device,
                    cfg=cfg,
                    tmax_n=tmax_n,
                    energy_weight=e_w,
                )

                val_data_total = val_data + float(cfg.alpha_v) * val_data_dv
                val_total = w_data_now * val_data_total + w_phys_now * val_phys

                history["val_total"].append(val_total)
                history["val_data"].append(val_data)
                history["val_data_dv"].append(val_data_dv)
                history["val_phys"].append(val_phys)

                # best checkpoint
                if val_total < best_val:
                    best_val = float(val_total)
                    extra_best = {
                        "train_config": asdict(cfg),
                        "scaler": scaler.to_dict(),
                        "model_config": _build_model_kwargs(cfg),
                        "history": history,
                        "best_val_total": best_val,
                        "best_epoch": epoch,
                        "weights": {"w_data": float(w_data_dyn), "w_phys": float(w_phys_dyn)},
                        "tmax_canonical": float(tmax_n),
                    }
                    save_checkpoint(best_path, model, extra=extra_best)

            # ---- Logging ----
            if (epoch % max(1, int(cfg.log_every)) == 0) or (epoch == adam_epochs - 1):
                msg = (
                    f"Epoch {epoch:4d}/{adam_epochs-1:4d} | "
                    f"Train: T={train_total:.3e} D={train_data:.3e} Dv={train_data_dv:.3e} P={train_phys:.3e} | "
                    f"Wdata={w_data_now:.3g} Wphys={w_phys_now:.3g} | "
                    f"tmax={(tmax_n * float(scaler.TU))/3600.0:.2f}h"
                )
                if val_loader is not None and len(history["val_total"]) > 0:
                    msg += f" | Val: T={val_total:.3e} D={val_data:.3e} Dv={val_data_dv:.3e} P={val_phys:.3e} | BestVal={best_val:.3e}"
                print(msg, flush=True)

            # ---- Periodic checkpoint ----
            if cfg.save_every and int(cfg.save_every) > 0 and (epoch % int(cfg.save_every) == 0) and epoch > 0:
                extra_mid = {
                    "train_config": asdict(cfg),
                    "scaler": scaler.to_dict(),
                    "model_config": _build_model_kwargs(cfg),
                    "history": history,
                    "best_val_total": best_val,
                    "best_epoch": epoch,
                    "weights": {"w_data": float(w_data_dyn), "w_phys": float(w_phys_dyn)},
                    "tmax_canonical": float(tmax_n),
                }
                save_checkpoint(ckpt_path, model, extra=extra_mid)

        # -----------------------
        # 2) L-BFGS fine-tune stage (optional)
        # -----------------------
        if cfg.use_lbfgs and int(cfg.lbfgs_max_iter) > 0:
            model.train()
            print(
                f"[INFO] L-BFGS fine-tune | max_iter={int(cfg.lbfgs_max_iter)} | sample={int(cfg.lbfgs_sample_size)}",
                flush=True,
            )

            sample_n = min(int(cfg.lbfgs_sample_size), len(train_ds))
            tmp_loader = DataLoader(
                train_ds,
                batch_size=min(int(cfg.batch_size), max(256, sample_n)),
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=pin_memory,
            )

            t_list: List[torch.Tensor] = []
            ic_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            got = 0
            for bt, bic, by in tmp_loader:
                take = min(int(bt.shape[0]), sample_n - got)
                t_list.append(bt[:take])
                ic_list.append(bic[:take])
                y_list.append(by[:take])
                got += int(take)
                if got >= sample_n:
                    break

            t_fixed = torch.cat(t_list, dim=0).to(device, non_blocking=True)
            ic_fixed = torch.cat(ic_list, dim=0).to(device, non_blocking=True)
            y_fixed = torch.cat(y_list, dim=0).to(device, non_blocking=True)

            tmax_n_final = float(dataset.t.max().item())

            opt_lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=float(cfg.lbfgs_lr),
                max_iter=int(cfg.lbfgs_max_iter),
                history_size=int(cfg.lbfgs_history_size),
                line_search_fn=(str(cfg.lbfgs_line_search) if cfg.lbfgs_line_search else None),
            )

            w_data_now = float(w_data_dyn)
            w_phys_now = float(w_phys_dyn)

            def closure():
                opt_lbfgs.zero_grad(set_to_none=True)

                bt, bic, by = t_fixed, ic_fixed, y_fixed
                bt, bic, by = _masked_batch(bt, bic, by, tmax_n=tmax_n_final, enabled=True)
                if bt.numel() == 0:
                    return torch.tensor(0.0, device=device, requires_grad=True)

                bt = bt.clone().detach().requires_grad_(True)
                yp = model(bt, bic)

                ld_mse = loss_fn(yp, by)
                drdt_p = d_dt(yp[:, 0:3], bt)

                ld_dv = torch.tensor(0.0, device=device)
                if cfg.use_derivative_data_loss and float(cfg.alpha_v) != 0.0:
                    ld_dv = torch.mean((drdt_p - by[:, 3:6]) ** 2)

                ld = ld_mse + float(cfg.alpha_v) * ld_dv

                e_w = float(cfg.energy_weight) if cfg.use_energy_loss else 0.0
                lp = _call_physics_loss(
                    model,
                    bt,
                    bic,
                    y_pred=yp,
                    drdt_pred=drdt_p,
                    energy_weight=e_w,
                )

                loss = w_data_now * ld + w_phys_now * lp
                loss.backward()
                return loss

            loss_lbfgs = opt_lbfgs.step(closure)
            try:
                loss_lbfgs_val = float(loss_lbfgs.item())
            except Exception:
                loss_lbfgs_val = float(loss_lbfgs)

            print(f"[OK] L-BFGS done | loss={loss_lbfgs_val:.3e}", flush=True)

        # -----------------------
        # Finalize outputs
        # -----------------------
        dt = time.time() - t0
        print(f"[OK] Training finished | elapsed={dt:.1f}s", flush=True)

        extra = {
            "train_config": asdict(cfg),
            "scaler": scaler.to_dict(),
            "model_config": _build_model_kwargs(cfg),
            "history": history,
            "best_val_total": best_val,
            "best_ckpt": os.path.basename(best_path) if os.path.exists(best_path) else None,
        }
        save_checkpoint(ckpt_path, model, extra=extra)

        # plot
        if cfg.save_plot:
            fig_path = os.path.join(cfg.out_dir, "loss_history.png")
            plt.figure(figsize=(8, 5))
            plt.plot(history["train_total"], label="train")
            if len(history["val_total"]) > 0:
                plt.plot(history["val_total"], label="val")
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.title("PINN Training Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            print(f"[OK] Loss fig: {fig_path}", flush=True)

        # meta json
        if cfg.save_meta:
            meta_path = os.path.join(cfg.out_dir, "train_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(extra, f, indent=2)
            print(f"[OK] Meta: {meta_path}", flush=True)

        print(f"[OK] Checkpoint: {ckpt_path}", flush=True)
        if os.path.exists(best_path):
            print(f"[OK] Best:       {best_path}", flush=True)

        return ckpt_path

    finally:
        # don't leak global dtype changes
        torch.set_default_dtype(prev_default_dtype)
