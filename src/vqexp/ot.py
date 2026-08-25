from __future__ import annotations

import math

import torch


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sq = (x * x).sum(dim=1, keepdim=True)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()
    return (x_sq + y_sq - 2.0 * (x @ y.t())).clamp_min(0.0)


@torch.no_grad()
def _sinkhorn_potentials(cost: torch.Tensor, eps: torch.Tensor, num_iters: int, tol: float):
    n, m = cost.shape
    log_a = -math.log(n)
    log_b = -math.log(m)
    f = cost.new_zeros(n)
    g = cost.new_zeros(m)

    for iteration in range(num_iters):
        f_prev = f
        g = -eps * torch.logsumexp((f.unsqueeze(1) - cost) / eps + log_a, dim=0)
        f = -eps * torch.logsumexp((g.unsqueeze(0) - cost) / eps + log_b, dim=1)
        if tol > 0 and (iteration + 1) % 5 == 0:
            if (f - f_prev).abs().max().item() < tol:
                break
    return f, g


@torch.no_grad()
def _sinkhorn_potentials_symmetric(
    cost: torch.Tensor,
    eps: torch.Tensor,
    num_iters: int,
    tol: float,
):
    n = cost.shape[0]
    log_a = -math.log(n)
    f = cost.new_zeros(n)

    for iteration in range(num_iters):
        f_prev = f
        transform = -eps * torch.logsumexp((f.unsqueeze(1) - cost) / eps + log_a, dim=0)
        f = 0.5 * (f + transform)
        if tol > 0 and (iteration + 1) % 5 == 0:
            if (f - f_prev).abs().max().item() < tol:
                break
    return f


def _entropic_ot_cost(
    cost: torch.Tensor,
    eps: torch.Tensor,
    num_iters: int,
    tol: float,
    symmetric: bool = False,
) -> torch.Tensor:
    n, m = cost.shape
    log_a = -math.log(n)
    log_b = -math.log(m)
    cost_detached = cost.detach()

    if symmetric:
        f = _sinkhorn_potentials_symmetric(cost_detached, eps, num_iters, tol)
        g = f
    else:
        f, g = _sinkhorn_potentials(cost_detached, eps, num_iters, tol)

    with torch.no_grad():
        log_plan = (f.unsqueeze(1) + g.unsqueeze(0) - cost_detached) / eps + log_a + log_b
        plan = log_plan.exp()
        plan = plan / plan.sum().clamp_min(1e-12)
    return (plan * cost).sum()


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    blur: float = 0.2,
    num_iters: int = 50,
    tol: float = 1e-6,
    eps_relative: bool = True,
    debias: bool = True,
) -> torch.Tensor:
    cost_xy = pairwise_sq_dist(x, y)
    if eps_relative:
        scale = cost_xy.detach().mean().clamp_min(1e-12)
        eps = (blur**2) * scale
    else:
        eps = torch.as_tensor(blur**2, device=cost_xy.device, dtype=cost_xy.dtype).clamp_min(1e-12)

    loss = _entropic_ot_cost(cost_xy, eps, num_iters, tol, symmetric=False)
    if debias:
        cost_xx = pairwise_sq_dist(x, x)
        cost_yy = pairwise_sq_dist(y, y)
        loss = loss - 0.5 * _entropic_ot_cost(cost_xx, eps, num_iters, tol, symmetric=True)
        loss = loss - 0.5 * _entropic_ot_cost(cost_yy, eps, num_iters, tol, symmetric=True)
    return loss


def _match_sorted_length(sorted_vals: torch.Tensor, k: int) -> torch.Tensor:
    n = sorted_vals.shape[0]
    if n == k:
        return sorted_vals
    pos = torch.linspace(0.0, n - 1.0, k, device=sorted_vals.device, dtype=sorted_vals.dtype)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    weight = (pos - lo.to(pos.dtype)).unsqueeze(1)
    return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 256,
    p: int = 2,
) -> torch.Tensor:
    dim = x.shape[1]
    theta = torch.randn(dim, num_projections, device=x.device, dtype=x.dtype)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)
    x_proj = x @ theta
    y_proj = y @ theta
    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)
    target_len = max(x_sorted.shape[0], y_sorted.shape[0])
    x_sorted = _match_sorted_length(x_sorted, target_len)
    y_sorted = _match_sorted_length(y_sorted, target_len)
    return ((x_sorted - y_sorted).abs() ** p).mean()


def sample_rows(tensor: torch.Tensor, count: int, replace: bool) -> torch.Tensor:
    n = tensor.shape[0]
    if count <= 0 or count >= n:
        return tensor
    if replace:
        indices = torch.randint(0, n, (count,), device=tensor.device)
    else:
        indices = torch.randperm(n, device=tensor.device)[:count]
    return tensor[indices]
