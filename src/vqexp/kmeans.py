from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def collect_latents_for_kmeans(model, train_loader, device: torch.device, max_samples: int) -> torch.Tensor:
    was_training = model.training
    model.eval()
    latent_chunks: list[torch.Tensor] = []
    collected = 0

    for images, _ in train_loader:
        images = images.to(device, non_blocking=True)
        ze = model.encode(images)
        ze_flat = ze.permute(0, 2, 3, 1).reshape(-1, ze.shape[1])

        remaining = max_samples - collected
        if remaining <= 0:
            break
        if ze_flat.shape[0] > remaining:
            selected = torch.randperm(ze_flat.shape[0], device=ze_flat.device)[:remaining]
            ze_flat = ze_flat[selected]

        latent_chunks.append(ze_flat.detach().float().cpu())
        collected += ze_flat.shape[0]
        if collected >= max_samples:
            break

    if was_training:
        model.train()
    if not latent_chunks:
        raise RuntimeError("K-means initialization failed: no latent vectors were collected.")

    latents = torch.cat(latent_chunks, dim=0)
    return latents[:max_samples]


def torch_kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    samples = samples.float()
    num_samples, embedding_dim = samples.shape
    if num_samples < num_clusters:
        raise ValueError(f"K-means needs at least {num_clusters} samples, got {num_samples}.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial_indices = torch.randperm(num_samples, generator=generator)[:num_clusters]
    centroids = samples[initial_indices].to(device)

    for iteration in range(num_iters):
        sums = torch.zeros(num_clusters, embedding_dim, device=device, dtype=torch.float32)
        counts = torch.zeros(num_clusters, device=device, dtype=torch.float32)

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            batch = samples[start:end].to(device, non_blocking=True)
            distances = (
                torch.sum(batch**2, dim=1, keepdim=True)
                + torch.sum(centroids**2, dim=1).unsqueeze(0)
                - 2.0 * batch @ centroids.t()
            )
            assignments = torch.argmin(distances, dim=1)
            sums.index_add_(0, assignments, batch)
            counts.index_add_(
                0,
                assignments,
                torch.ones(assignments.shape[0], device=device, dtype=torch.float32),
            )

        non_empty = counts > 0
        new_centroids = centroids.clone()
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)
        empty_indices = torch.nonzero(~non_empty, as_tuple=False).flatten()
        if empty_indices.numel() > 0:
            replacement_indices = torch.randint(0, num_samples, (empty_indices.numel(),), generator=generator)
            new_centroids[empty_indices] = samples[replacement_indices].to(device)

        shift = torch.mean(torch.norm(new_centroids - centroids, dim=1)).item()
        centroids = new_centroids
        print(f"  [PyTorch K-means {iteration + 1:02d}/{num_iters:02d}] center_shift={shift:.6f}")

    return centroids.detach().cpu()


def run_kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    num_samples = samples.shape[0]
    if num_samples < num_clusters:
        raise ValueError(f"K-means needs at least {num_clusters} samples, got {num_samples}.")

    try:
        from sklearn.cluster import MiniBatchKMeans

        print(f"Using sklearn MiniBatchKMeans: N={num_samples}, K={num_clusters}, D={samples.shape[1]}")
        batch_size = min(int(cfg["kmeans_batch_size"]), num_samples)
        batch_size = max(batch_size, min(num_clusters, num_samples))
        init_size = min(num_samples, max(3 * num_clusters, batch_size))
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=1,
            max_iter=int(cfg["kmeans_iters"]),
            batch_size=batch_size,
            init_size=init_size,
            random_state=int(cfg["seed"]),
            reassignment_ratio=0.01,
            compute_labels=False,
            verbose=0,
        )
        kmeans.fit(samples.numpy())
        return torch.from_numpy(kmeans.cluster_centers_).float()
    except ImportError:
        print("sklearn is unavailable; falling back to chunked PyTorch K-means.")
        return torch_kmeans(
            samples=samples,
            num_clusters=num_clusters,
            num_iters=int(cfg["kmeans_iters"]),
            chunk_size=int(cfg["kmeans_chunk_size"]),
            device=device,
            seed=int(cfg["seed"]),
        )


@torch.no_grad()
def initialize_codebook_with_kmeans(
    model,
    train_loader,
    device: torch.device,
    cfg: dict[str, Any],
    optimizer=None,
) -> None:
    num_embeddings = model.vq_layer.num_embeddings
    max_samples = max(int(cfg["kmeans_num_samples"]), num_embeddings)
    print(f"[Init] Collecting {max_samples} latents for K-means...")
    latent_samples = collect_latents_for_kmeans(model, train_loader, device, max_samples)
    print(f"[Init] Collected latent samples: {tuple(latent_samples.shape)}")

    centroids = run_kmeans(latent_samples, num_embeddings, cfg, device)
    model.vq_layer.copy_base_codebook(centroids)

    if optimizer is not None:
        for parameter in model.vq_layer.embedding.parameters():
            optimizer.state.pop(parameter, None)
    print("[Init] K-means codebook initialization complete.")
