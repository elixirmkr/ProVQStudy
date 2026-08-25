from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import write_json
from .cov_init import initialize_codebook_proj_with_latent_covariance
from .data import get_dataloaders
from .kmeans import initialize_codebook_with_kmeans
from .metrics import LatentEffectiveRankTracker, calculate_psnr, calculate_ssim
from .models import VQVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_dir_for(cfg: dict[str, Any]) -> Path:
    return (
        Path(cfg["output_dir"])
        / cfg["suite_name"]
        / cfg["experiment_name"]
        / f"seed_{int(cfg['seed'])}"
    )


def get_provq_schedule(
    epoch: int,
    batch_idx: int,
    num_batches: int,
    warmup_epochs: int,
    transition_epochs: int,
    lambda_initial: float,
) -> dict[str, Any]:
    if epoch <= warmup_epochs:
        return {"stage": "warmup", "continuous": True, "alpha": 1.0, "omega": 0.0}

    if transition_epochs <= 0:
        return {"stage": "hard_vq", "continuous": False, "alpha": 0.0, "omega": 1.0}

    transition_step = (epoch - warmup_epochs - 1) * num_batches + batch_idx
    total_transition_steps = transition_epochs * num_batches
    if transition_step >= total_transition_steps:
        return {"stage": "hard_vq", "continuous": False, "alpha": 0.0, "omega": 1.0}

    if total_transition_steps <= 1:
        progress = 1.0
    else:
        progress = transition_step / (total_transition_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
    alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
    omega = lambda_initial + (1.0 - lambda_initial) * (1.0 - alpha)
    return {"stage": "transition", "continuous": False, "alpha": alpha, "omega": omega}


def dist_match_lr_mult(
    cfg: dict[str, Any],
    epoch: int,
    batch_idx: int,
    num_batches: int,
) -> float:
    if not cfg.get("dist_match_enabled", False):
        return 0.0

    warmup_epochs = int(cfg["provq_warmup_epochs"])
    if epoch <= warmup_epochs:
        return float(cfg["dist_match_warmup_lr_mult"])

    if cfg["dist_match_schedule"] == "warmup_only":
        return 0.0

    total_stage2_steps = max((int(cfg["epochs"]) - warmup_epochs) * num_batches, 1)
    current = (epoch - warmup_epochs - 1) * num_batches + batch_idx
    if total_stage2_steps <= 1:
        progress = 1.0
    else:
        progress = min(max(current / (total_stage2_steps - 1), 0.0), 1.0)

    start = float(cfg["dist_match_stage2_lr_mult_start"])
    end = float(cfg["dist_match_stage2_lr_mult_end"])
    if cfg["dist_match_anneal_kind"] == "linear":
        factor = 1.0 - progress
    else:
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * factor


@torch.no_grad()
def evaluate_model(model: VQVAE, test_loader, device: torch.device, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    all_indices = []
    tracker = LatentEffectiveRankTracker(int(cfg["embedding_dim"]), device)

    for images, _ in test_loader:
        images = images.to(device, non_blocking=True)
        x_recon, ze, _, _, indices = model(images, alpha=0.0, continuous=False)
        total_psnr += calculate_psnr(images, x_recon)
        total_ssim += calculate_ssim(images, x_recon)
        all_indices.append(indices.reshape(-1).cpu())
        ze_flat = ze.permute(0, 2, 3, 1).reshape(-1, ze.shape[1])
        tracker.update(ze_flat)

    all_indices_t = torch.cat(all_indices)
    counts = torch.bincount(all_indices_t, minlength=int(cfg["num_embeddings"]))
    probs = counts.float() / counts.sum().clamp_min(1)
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))).item()
    codebook_usage = (counts > 0).sum().item() / int(cfg["num_embeddings"]) * 100.0
    eff_rank, norm_eff_rank = tracker.compute()

    codebook = model.vq_layer.get_codebook()
    norm_sq = (codebook**2).sum(dim=1, keepdim=True)
    dist_sq = (norm_sq + norm_sq.t() - 2.0 * (codebook @ codebook.t())).clamp_min(0.0)
    pairwise_distance = torch.sqrt(dist_sq)
    triu = torch.triu_indices(codebook.shape[0], codebook.shape[0], offset=1, device=codebook.device)
    avg_pairwise_dist = pairwise_distance[triu[0], triu[1]].mean().item()

    return {
        "psnr": total_psnr / len(test_loader),
        "ssim": total_ssim / len(test_loader),
        "perplexity": perplexity,
        "usage": codebook_usage,
        "effective_rank": eff_rank,
        "normalized_effective_rank": norm_eff_rank,
        "avg_pairwise_distance": avg_pairwise_dist,
    }


def build_main_optimizer(model: VQVAE, cfg: dict[str, Any]):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(params, lr=float(cfg["lr"]))


def build_dist_optimizer(model: VQVAE, cfg: dict[str, Any]):
    if not cfg.get("dist_match_enabled", False):
        return None
    params = [parameter for parameter in model.vq_layer.trainable_codebook_parameters() if parameter.requires_grad]
    if not params:
        return None
    return torch.optim.Adam(params, lr=float(cfg["lr"]) * float(cfg["dist_match_warmup_lr_mult"]))


def train_one_epoch(
    model: VQVAE,
    train_loader,
    main_optimizer,
    dist_optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float | str]:
    model.train()
    totals = {
        "recon": 0.0,
        "codebook": 0.0,
        "commit": 0.0,
        "weighted_vq": 0.0,
        "dist_match": 0.0,
        "dist_match_lr_mult": 0.0,
        "total": 0.0,
        "alpha": 0.0,
        "omega": 0.0,
    }
    num_batches = len(train_loader)
    last_stage = "unknown"

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        schedule = get_provq_schedule(
            epoch=epoch,
            batch_idx=batch_idx,
            num_batches=num_batches,
            warmup_epochs=int(cfg["provq_warmup_epochs"]),
            transition_epochs=int(cfg["provq_transition_epochs"]),
            lambda_initial=float(cfg["provq_lambda"]),
        )
        stage = schedule["stage"]
        continuous = bool(schedule["continuous"])
        alpha = float(schedule["alpha"])
        omega = float(schedule["omega"])
        last_stage = stage

        query_codebook = continuous and bool(cfg["warmup_codebook_loss"])
        main_optimizer.zero_grad(set_to_none=True)
        x_recon, ze, zq, _, _ = model(
            images,
            alpha=alpha,
            continuous=continuous,
            query_codebook=query_codebook,
        )
        recon_loss = F.mse_loss(x_recon, images)
        codebook_loss = ze.new_zeros(())
        commit_loss = ze.new_zeros(())
        weighted_vq_loss = ze.new_zeros(())
        main_loss = recon_loss

        if continuous:
            if cfg["warmup_codebook_loss"]:
                codebook_loss = F.mse_loss(zq, ze.detach())
                main_loss = main_loss + float(cfg["warmup_codebook_weight"]) * codebook_loss
        else:
            codebook_loss = F.mse_loss(zq, ze.detach())
            commit_loss = F.mse_loss(ze, zq.detach())
            vq_loss = codebook_loss + float(cfg["beta_commit"]) * commit_loss
            weighted_vq_loss = omega * vq_loss
            main_loss = main_loss + weighted_vq_loss

        ze_for_dist = ze.detach()
        main_loss.backward()
        main_optimizer.step()

        dist_loss = ze.new_zeros(())
        current_dist_mult = dist_match_lr_mult(cfg, epoch, batch_idx, num_batches)
        if dist_optimizer is not None and current_dist_mult > 0.0:
            for group in dist_optimizer.param_groups:
                group["lr"] = float(cfg["lr"]) * current_dist_mult
            dist_optimizer.zero_grad(set_to_none=True)
            dist_loss = model.codebook_distribution_matching_loss(
                ze_for_dist,
                method=cfg["dist_match_method"],
                latent_samples=int(cfg["dist_match_latent_samples"]),
                codebook_samples=int(cfg["dist_match_codebook_samples"]),
                sinkhorn_blur=float(cfg["dist_match_sinkhorn_blur"]),
                sinkhorn_iters=int(cfg["dist_match_sinkhorn_iters"]),
                sinkhorn_tol=float(cfg["dist_match_sinkhorn_tol"]),
                sinkhorn_eps_relative=bool(cfg["dist_match_sinkhorn_eps_relative"]),
                sinkhorn_debias=bool(cfg["dist_match_sinkhorn_debias"]),
                sw_projections=int(cfg["dist_match_sw_projections"]),
            )
            dist_loss.backward()
            dist_optimizer.step()

        total_loss = main_loss.detach() + dist_loss.detach()
        totals["recon"] += recon_loss.item()
        totals["codebook"] += codebook_loss.item()
        totals["commit"] += commit_loss.item()
        totals["weighted_vq"] += weighted_vq_loss.item()
        totals["dist_match"] += dist_loss.item()
        totals["dist_match_lr_mult"] += current_dist_mult
        totals["total"] += total_loss.item()
        totals["alpha"] += alpha
        totals["omega"] += omega

    stats = {key: value / num_batches for key, value in totals.items()}
    stats["stage"] = last_stage
    return stats


def maybe_initialize_after_warmup(
    model: VQVAE,
    train_loader,
    device: torch.device,
    cfg: dict[str, Any],
    optimizer,
    initialized: bool,
) -> bool:
    if initialized:
        return True
    if cfg["init_method"] == "kmeans":
        initialize_codebook_with_kmeans(model, train_loader, device, cfg, optimizer)
        return True
    if cfg["init_method"] == "cov":
        initialize_codebook_proj_with_latent_covariance(model, train_loader, device, cfg, optimizer)
        return True
    return initialized


def save_checkpoint(path: Path, model: VQVAE, main_optimizer, dist_optimizer, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "main_optimizer": main_optimizer.state_dict(),
        "cfg": cfg,
        **extra,
    }
    if dist_optimizer is not None:
        payload["dist_optimizer"] = dist_optimizer.state_dict()
    torch.save(payload, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")


def print_epoch(epoch: int, stats: dict[str, Any], metrics: dict[str, float] | None) -> None:
    parts = [
        f"[Epoch {epoch:03d}]",
        f"stage={stats['stage']:<10s}",
        f"alpha={stats['alpha']:.4f}",
        f"omega={stats['omega']:.4f}",
        f"loss_tot={stats['total']:.4f}",
        f"recon={stats['recon']:.4f}",
        f"codebook={stats['codebook']:.4f}",
        f"commit={stats['commit']:.4f}",
        f"weighted_vq={stats['weighted_vq']:.4f}",
        f"dist_match={stats['dist_match']:.4f}",
        f"dist_lr_mult={stats['dist_match_lr_mult']:.4f}",
    ]
    if metrics is not None:
        parts.extend(
            [
                f"PSNR={metrics['psnr']:.2f}",
                f"SSIM={metrics['ssim']:.4f}",
                f"PPLX={metrics['perplexity']:.2f}",
                f"usage={metrics['usage']:.1f}%",
                f"EffRank={metrics['effective_rank']:.2f}",
                f"NormEffRank={metrics['normalized_effective_rank']:.4f}",
                f"AvgPairDist={metrics['avg_pairwise_distance']:.4f}",
            ]
        )
    print(" | ".join(parts), flush=True)


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    print(
        f"Starting {cfg['suite_name']}/{cfg['experiment_name']} seed={cfg['seed']} "
        f"on device={device} output={run_dir}",
        flush=True,
    )
    train_loader, test_loader = get_dataloaders(cfg)
    model = VQVAE(
        in_channels=3,
        embedding_dim=int(cfg["embedding_dim"]),
        num_embeddings=int(cfg["num_embeddings"]),
        quantizer=cfg["quantizer"],
        simvq_proj_init=cfg["simvq_proj_init"],
    ).to(device)

    main_optimizer = build_main_optimizer(model, cfg)
    dist_optimizer = build_dist_optimizer(model, cfg)

    initialized = False
    if int(cfg["provq_warmup_epochs"]) == 0:
        initialized = maybe_initialize_after_warmup(
            model,
            train_loader,
            device,
            cfg,
            main_optimizer,
            initialized,
        )

    best_psnr = -float("inf")
    best_epoch = None
    final_metrics: dict[str, float] | None = None
    start_time = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        stats = train_one_epoch(
            model,
            train_loader,
            main_optimizer,
            dist_optimizer,
            device,
            cfg,
            epoch,
        )

        if int(cfg["provq_warmup_epochs"]) > 0 and epoch == int(cfg["provq_warmup_epochs"]):
            initialized = maybe_initialize_after_warmup(
                model,
                train_loader,
                device,
                cfg,
                main_optimizer,
                initialized,
            )

        should_eval = epoch % int(cfg["eval_every"]) == 0 or epoch == int(cfg["epochs"])
        metrics = evaluate_model(model, test_loader, device, cfg) if should_eval else None
        if metrics is not None:
            final_metrics = metrics

        print_epoch(epoch, stats, metrics)
        record: dict[str, Any] = {"epoch": epoch, "train": stats}
        if metrics is not None:
            record["eval"] = metrics
        append_jsonl(metrics_path, record)

        quantizer_ready = int(cfg["provq_warmup_epochs"]) == 0 or epoch >= int(cfg["provq_warmup_epochs"])
        if metrics is not None and quantizer_ready and metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                model,
                main_optimizer,
                dist_optimizer,
                cfg,
                {
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "initialized": initialized,
                },
            )

        ckpt_every = int(cfg["checkpoint_every"])
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:03d}.pt",
                model,
                main_optimizer,
                dist_optimizer,
                cfg,
                {"epoch": epoch, "best_psnr": best_psnr, "initialized": initialized},
            )

    save_checkpoint(
        run_dir / "last.pt",
        model,
        main_optimizer,
        dist_optimizer,
        cfg,
        {"epoch": int(cfg["epochs"]), "best_psnr": best_psnr, "initialized": initialized},
    )

    elapsed = time.time() - start_time
    summary = {
        "suite_name": cfg["suite_name"],
        "experiment_name": cfg["experiment_name"],
        "experiment_label": cfg["experiment_label"],
        "group": cfg["group"],
        "seed": int(cfg["seed"]),
        "best_psnr": best_psnr,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
        "elapsed_seconds": elapsed,
        "run_dir": os.fspath(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"Finished {cfg['experiment_name']} seed={cfg['seed']} | "
        f"best_psnr={best_psnr:.2f} | best_epoch={best_epoch} | elapsed={elapsed / 60:.1f} min",
        flush=True,
    )
    return summary
