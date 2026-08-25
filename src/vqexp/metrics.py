from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim_loss


MEAN_T = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
STD_T = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


def denormalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    mean = MEAN_T.view(1, 3, 1, 1).to(tensor.device)
    std = STD_T.view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    img1 = denormalize_to_01(original).detach().cpu().numpy()
    img2 = denormalize_to_01(reconstructed).detach().cpu().numpy()
    total = 0.0
    for index in range(img1.shape[0]):
        x = np.transpose(img1[index], (1, 2, 0))
        y = np.transpose(img2[index], (1, 2, 0))
        value = peak_signal_noise_ratio(x, y, data_range=1.0)
        if np.isinf(value):
            value = 100.0
        total += value
    return float(total / img1.shape[0])


def calculate_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    img1 = denormalize_to_01(original).detach().cpu().numpy()
    img2 = denormalize_to_01(reconstructed).detach().cpu().numpy()
    total = 0.0
    for index in range(img1.shape[0]):
        x = np.transpose(img1[index], (1, 2, 0))
        y = np.transpose(img2[index], (1, 2, 0))
        try:
            value = ssim_loss(x, y, channel_axis=2, data_range=1.0)
        except TypeError:
            value = ssim_loss(x, y, multichannel=True, data_range=1.0)
        total += value
    return float(total / img1.shape[0])


class LatentEffectiveRankTracker:
    def __init__(self, dim: int, device: torch.device):
        self.dim = dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum_vec = torch.zeros(self.dim, device=self.device, dtype=torch.float64)
        self.sum_outer = torch.zeros(self.dim, self.dim, device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, z_flat: torch.Tensor) -> None:
        z_flat = z_flat.detach().to(torch.float64)
        self.count += z_flat.shape[0]
        self.sum_vec += z_flat.sum(dim=0)
        self.sum_outer += z_flat.t() @ z_flat

    @torch.no_grad()
    def compute(self, eps: float = 1e-12) -> tuple[float, float]:
        if self.count < 2:
            return 0.0, 0.0
        mean = self.sum_vec / self.count
        cov = self.sum_outer / self.count - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.t())
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        total = eigenvalues.sum()
        if total.item() < eps:
            return 0.0, 0.0
        probs = eigenvalues / total
        nonzero_probs = probs[probs > eps]
        entropy = -torch.sum(nonzero_probs * torch.log(nonzero_probs))
        effective_rank = torch.exp(entropy).item()
        return effective_rank, effective_rank / self.dim
