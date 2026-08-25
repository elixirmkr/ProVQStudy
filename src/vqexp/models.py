from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot import sample_rows, sinkhorn_divergence, sliced_wasserstein_distance


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, embedding_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embedding_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, embedding_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        kind: str = "vanilla",
        simvq_proj_init: str = "identity",
    ):
        super().__init__()
        if kind not in {"vanilla", "simvq"}:
            raise ValueError(f"Unknown quantizer kind: {kind}")
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.kind = kind

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_dim**-0.5)

        self.embedding_proj: nn.Linear | None = None
        if kind == "simvq":
            for parameter in self.embedding.parameters():
                parameter.requires_grad = False
            self.embedding_proj = nn.Linear(embedding_dim, embedding_dim)
            if simvq_proj_init == "identity":
                nn.init.eye_(self.embedding_proj.weight)
                nn.init.zeros_(self.embedding_proj.bias)
            elif simvq_proj_init != "default":
                raise ValueError(f"Unknown SimVQ projection init: {simvq_proj_init}")

    def get_codebook(self) -> torch.Tensor:
        if self.embedding_proj is None:
            return self.embedding.weight
        return self.embedding_proj(self.embedding.weight)

    def trainable_codebook_parameters(self) -> Iterable[nn.Parameter]:
        if self.embedding_proj is None:
            return self.embedding.parameters()
        return self.embedding_proj.parameters()

    @torch.no_grad()
    def copy_base_codebook(self, centroids: torch.Tensor) -> None:
        expected = self.embedding.weight.shape
        if centroids.shape != expected:
            raise RuntimeError(f"Centroid shape {tuple(centroids.shape)} != expected {tuple(expected)}")
        self.embedding.weight.copy_(centroids.to(self.embedding.weight.device, self.embedding.weight.dtype))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_ = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_.reshape(-1, self.embedding_dim)
        quant_codebook = self.get_codebook()
        flat_for_search = flat_input.detach()

        distances = (
            torch.sum(flat_for_search**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1).unsqueeze(0)
            - 2.0 * torch.matmul(flat_for_search, quant_codebook.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, quant_codebook)
        quantized = quantized.reshape(inputs_.shape).permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.reshape(inputs_.shape[:-1])
        return quantized, encoding_indices, distances

    def distribution_matching_loss(
        self,
        ze: torch.Tensor,
        method: str = "sinkhorn",
        latent_samples: int = 1024,
        codebook_samples: int = 1024,
        sinkhorn_blur: float = 0.2,
        sinkhorn_iters: int = 50,
        sinkhorn_tol: float = 1e-6,
        sinkhorn_eps_relative: bool = True,
        sinkhorn_debias: bool = True,
        sw_projections: int = 256,
    ) -> torch.Tensor:
        z = ze.detach().permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        z = sample_rows(z, latent_samples, replace=True)
        e = sample_rows(self.get_codebook(), codebook_samples, replace=False)
        if method == "sinkhorn":
            return sinkhorn_divergence(
                z,
                e,
                blur=sinkhorn_blur,
                num_iters=sinkhorn_iters,
                tol=sinkhorn_tol,
                eps_relative=sinkhorn_eps_relative,
                debias=sinkhorn_debias,
            )
        if method == "sliced_wasserstein":
            return sliced_wasserstein_distance(z, e, num_projections=sw_projections)
        raise ValueError(f"Unknown distribution matching method: {method}")


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 32,
        num_embeddings: int = 1024,
        quantizer: str = "vanilla",
        simvq_proj_init: str = "identity",
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, quantizer, simvq_proj_init)
        self.decoder = Decoder(in_channels, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize_latent(self, ze: torch.Tensor):
        return self.vq_layer(ze)

    def quantize(self, x: torch.Tensor):
        ze = self.encoder(x)
        zq, indices, distances = self.vq_layer(ze)
        return ze, zq, indices, distances

    def codebook_distribution_matching_loss(self, ze: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.vq_layer.distribution_matching_loss(ze, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 0.0,
        continuous: bool = False,
        query_codebook: bool = False,
    ):
        ze = self.encoder(x)
        if continuous and not query_codebook:
            return self.decoder(ze), ze, None, None, None

        zq, indices, distances = self.vq_layer(ze)
        if continuous:
            return self.decoder(ze), ze, zq, distances, indices

        zq_ste = ze + (zq - ze).detach()
        z_tilde = float(alpha) * ze + (1.0 - float(alpha)) * zq_ste
        return self.decoder(z_tilde), ze, zq, distances, indices
