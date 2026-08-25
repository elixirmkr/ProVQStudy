from __future__ import annotations

from typing import Any

import torch


def symmetric_eigh(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mat = 0.5 * (mat + mat.t())
    return torch.linalg.eigh(mat)


def symmetric_matrix_power(mat: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    evals, evecs = symmetric_eigh(mat)
    evals = torch.clamp(evals, min=0.0) + eps
    scaled = evals.pow(power)
    return (evecs * scaled.unsqueeze(0)) @ evecs.t()


def covariance_effective_rank(cov: torch.Tensor, eps: float = 1e-12) -> tuple[float, float]:
    evals, _ = symmetric_eigh(cov)
    evals = torch.clamp(evals, min=0.0)
    total = evals.sum()
    if total <= eps:
        return 0.0, 0.0
    probs = evals / total
    entropy = -(probs * torch.log(probs + eps)).sum()
    eff_rank = torch.exp(entropy).item()
    return eff_rank, eff_rank / cov.shape[0]


@torch.no_grad()
def estimate_latent_statistics(
    model,
    train_loader,
    device: torch.device,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    was_training = model.training
    model.eval()
    embedding_dim = model.vq_layer.embedding_dim
    sum_vec = torch.zeros(embedding_dim, dtype=torch.float64, device=device)
    sum_outer = torch.zeros(embedding_dim, embedding_dim, dtype=torch.float64, device=device)
    count = 0

    for images, _ in train_loader:
        if count >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        ze = model.encode(images)
        ze_flat = ze.permute(0, 2, 3, 1).reshape(-1, ze.shape[1])

        remaining = max_samples - count
        if ze_flat.shape[0] > remaining:
            selected = torch.randperm(ze_flat.shape[0], device=ze_flat.device)[:remaining]
            ze_flat = ze_flat[selected]

        ze64 = ze_flat.detach().to(torch.float64)
        sum_vec += ze64.sum(dim=0)
        sum_outer += ze64.t() @ ze64
        count += ze64.shape[0]

    if was_training:
        model.train()
    if count < 2:
        raise RuntimeError("Covariance initialization failed: not enough latent vectors.")

    mean = sum_vec / count
    cov = sum_outer / count - torch.outer(mean, mean)
    cov = cov * (count / (count - 1))
    cov = 0.5 * (cov + cov.t())
    return mean.cpu(), cov.cpu(), count


@torch.no_grad()
def initialize_codebook_proj_with_latent_covariance(
    model,
    train_loader,
    device: torch.device,
    cfg: dict[str, Any],
    optimizer=None,
) -> dict[str, float]:
    if model.vq_layer.embedding_proj is None:
        raise ValueError("Covariance initialization requires a SimVQ projection layer.")

    print(f"[Init] Estimating latent statistics for covariance init (max={cfg['cov_num_samples']})...")
    mean, cov, count = estimate_latent_statistics(model, train_loader, device, int(cfg["cov_num_samples"]))
    if count < 10 * int(cfg["embedding_dim"]):
        print(f"[Init] Warning: only {count} latent samples for dim={cfg['embedding_dim']}.")

    info = init_proj_from_latent_statistics(
        vq_layer=model.vq_layer,
        latent_mean=mean,
        latent_cov=cov,
        eps=float(cfg["cov_init_eps"]),
        shrinkage=float(cfg["cov_init_shrinkage"]),
        scale=float(cfg["cov_init_scale"]),
    )

    if optimizer is not None:
        for parameter in model.vq_layer.embedding_proj.parameters():
            optimizer.state.pop(parameter, None)

    print(
        "[Init] Covariance init complete | "
        f"samples={count} | latent_eff_rank={info['latent_eff_rank']:.2f} | "
        f"mean_err={info['codebook_mean_error']:.3e} | "
        f"cov_rel_err={info['codebook_cov_rel_error']:.3e}"
    )
    return info


@torch.no_grad()
def init_proj_from_latent_statistics(
    vq_layer,
    latent_mean: torch.Tensor,
    latent_cov: torch.Tensor,
    eps: float = 1e-6,
    shrinkage: float = 0.0,
    scale: float = 1.0,
) -> dict[str, float]:
    param = vq_layer.embedding_proj.weight
    device = param.device
    embedding_dim = vq_layer.embedding_dim

    mu = latent_mean.to(device=device, dtype=torch.float64).reshape(-1)
    sigma = latent_cov.to(device=device, dtype=torch.float64)
    if mu.numel() != embedding_dim or sigma.shape != (embedding_dim, embedding_dim):
        raise ValueError("Latent statistics have incompatible shapes.")

    sigma = 0.5 * (sigma + sigma.t())
    if shrinkage > 0.0:
        avg_var = torch.diagonal(sigma).mean()
        identity = torch.eye(embedding_dim, device=device, dtype=torch.float64)
        sigma = (1.0 - shrinkage) * sigma + shrinkage * avg_var * identity

    base = vq_layer.embedding.weight.detach().to(device=device, dtype=torch.float64)
    base_mean = base.mean(dim=0)
    base_centered = base - base_mean
    base_cov = base_centered.t() @ base_centered / max(base.shape[0] - 1, 1)
    base_cov = 0.5 * (base_cov + base_cov.t())

    sigma_sqrt = symmetric_matrix_power(sigma, 0.5, eps)
    base_inv_sqrt = symmetric_matrix_power(base_cov, -0.5, eps)
    weight = float(scale) * (sigma_sqrt @ base_inv_sqrt)
    bias = mu - weight @ base_mean

    vq_layer.embedding_proj.weight.copy_(weight.to(param.dtype))
    vq_layer.embedding_proj.bias.copy_(bias.to(param.dtype))

    codebook = vq_layer.get_codebook().detach().to(torch.float64)
    cb_mean = codebook.mean(dim=0)
    cb_centered = codebook - cb_mean
    cb_cov = cb_centered.t() @ cb_centered / max(codebook.shape[0] - 1, 1)
    target_cov = (float(scale) ** 2) * sigma
    mean_err = torch.norm(cb_mean - mu).item()
    cov_err = (torch.norm(cb_cov - target_cov) / torch.norm(target_cov).clamp_min(1e-12)).item()
    latent_eff_rank, latent_norm_eff_rank = covariance_effective_rank(sigma)
    evals, _ = symmetric_eigh(sigma)
    evals = torch.clamp(evals, min=0.0)

    return {
        "latent_mean_norm": torch.norm(mu).item(),
        "latent_trace": torch.diagonal(sigma).sum().item(),
        "latent_eig_max": evals.max().item(),
        "latent_eig_min": evals.min().item(),
        "latent_eff_rank": latent_eff_rank,
        "latent_norm_eff_rank": latent_norm_eff_rank,
        "codebook_mean_error": mean_err,
        "codebook_cov_rel_error": cov_err,
        "proj_weight_norm": torch.norm(weight).item(),
    }
