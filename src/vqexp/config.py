from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "suite_name": "manual",
    "experiment_name": "manual",
    "experiment_label": "manual",
    "group": "manual",
    "dataset": "cifar10",
    "data_root": "./data",
    "output_dir": "./runs",
    "train_subset": 0,
    "test_subset": 0,
    "batch_size": 128,
    "num_workers": 4,
    "epochs": 100,
    "eval_every": 1,
    "checkpoint_every": 0,
    "lr": 2e-4,
    "beta_commit": 0.25,
    "num_embeddings": 8192,
    "embedding_dim": 32,
    "seed": 42,
    "device": "auto",
    "quantizer": "vanilla",
    "simvq_proj_init": "identity",
    "provq_warmup_epochs": 0,
    "provq_transition_epochs": 0,
    "provq_lambda": 0.1,
    "warmup_codebook_loss": False,
    "warmup_codebook_weight": 1.0,
    "init_method": "none",
    "kmeans_num_samples": 100000,
    "kmeans_batch_size": 16384,
    "kmeans_iters": 20,
    "kmeans_chunk_size": 1024,
    "cov_num_samples": 200000,
    "cov_init_eps": 1e-6,
    "cov_init_shrinkage": 0.0,
    "cov_init_scale": 1.0,
    "dist_match_enabled": False,
    "dist_match_method": "sliced_wasserstein",
    "dist_match_schedule": "warmup_only",
    "dist_match_latent_samples": 2048,
    "dist_match_codebook_samples": 2048,
    "dist_match_sinkhorn_blur": 0.2,
    "dist_match_sinkhorn_iters": 50,
    "dist_match_sinkhorn_tol": 1e-6,
    "dist_match_sinkhorn_eps_relative": True,
    "dist_match_sinkhorn_debias": True,
    "dist_match_sw_projections": 256,
    "dist_match_warmup_lr_mult": 5.0,
    "dist_match_stage2_lr_mult_start": 5.0,
    "dist_match_stage2_lr_mult_end": 0.0,
    "dist_match_anneal_kind": "cosine",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_overrides(assignments: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not assignments:
        return overrides
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        overrides[key] = parse_scalar(raw_value)
    return overrides


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = load_json(path)
    if "experiments" not in suite or not isinstance(suite["experiments"], list):
        raise ValueError("Suite file must contain an experiments list.")
    return suite


def experiment_names(suite: dict[str, Any], group: str | None = None) -> list[str]:
    names: list[str] = []
    for experiment in suite["experiments"]:
        if group and experiment.get("group", "core") != group:
            continue
        names.append(experiment["name"])
    return names


def find_experiment(suite: dict[str, Any], name: str) -> dict[str, Any]:
    for experiment in suite["experiments"]:
        if experiment["name"] == name:
            return experiment
    names = ", ".join(experiment_names(suite))
    raise KeyError(f"Unknown experiment '{name}'. Available: {names}")


def materialize_config(
    suite_path: str | Path,
    experiment_name: str,
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    experiment = find_experiment(suite, experiment_name)
    cfg = deep_update(DEFAULT_CONFIG, suite.get("defaults", {}))
    cfg = deep_update(cfg, experiment.get("overrides", {}))
    cfg["suite_name"] = suite.get("name", Path(suite_path).stem)
    cfg["experiment_name"] = experiment["name"]
    cfg["experiment_label"] = experiment.get("label", experiment["name"])
    cfg["group"] = experiment.get("group", "core")
    if seed is not None:
        cfg["seed"] = int(seed)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg["dataset"] not in {"cifar10", "fake"}:
        raise ValueError("dataset must be 'cifar10' or 'fake'.")
    if cfg["quantizer"] not in {"vanilla", "simvq"}:
        raise ValueError("quantizer must be 'vanilla' or 'simvq'.")
    if cfg["simvq_proj_init"] not in {"identity", "default"}:
        raise ValueError("simvq_proj_init must be 'identity' or 'default'.")
    if cfg["init_method"] not in {"none", "kmeans", "cov"}:
        raise ValueError("init_method must be 'none', 'kmeans', or 'cov'.")
    if cfg["init_method"] == "cov" and cfg["quantizer"] != "simvq":
        raise ValueError("cov initialization is defined for the SimVQ projection.")
    if cfg["provq_warmup_epochs"] < 0:
        raise ValueError("provq_warmup_epochs cannot be negative.")
    if cfg["provq_transition_epochs"] < 0:
        raise ValueError("provq_transition_epochs cannot be negative.")
    if cfg["provq_warmup_epochs"] >= cfg["epochs"]:
        raise ValueError("provq_warmup_epochs must be smaller than epochs.")
    if not 0.0 <= cfg["provq_lambda"] <= 1.0:
        raise ValueError("provq_lambda must be in [0, 1].")
    if cfg["warmup_codebook_loss"] and cfg["warmup_codebook_weight"] <= 0:
        raise ValueError("warmup_codebook_weight must be positive.")
    if cfg["init_method"] == "kmeans" and cfg["kmeans_num_samples"] < cfg["num_embeddings"]:
        raise ValueError("kmeans_num_samples must be >= num_embeddings.")
    if cfg["init_method"] == "cov" and cfg["cov_num_samples"] < cfg["embedding_dim"] + 1:
        raise ValueError("cov_num_samples must be > embedding_dim.")
    if not 0.0 <= cfg["cov_init_shrinkage"] < 1.0:
        raise ValueError("cov_init_shrinkage must be in [0, 1).")
    if cfg["cov_init_eps"] <= 0.0 or cfg["cov_init_scale"] <= 0.0:
        raise ValueError("cov_init_eps and cov_init_scale must be positive.")
    if cfg["dist_match_method"] not in {"sinkhorn", "sliced_wasserstein"}:
        raise ValueError("dist_match_method must be 'sinkhorn' or 'sliced_wasserstein'.")
    if cfg["dist_match_schedule"] not in {"warmup_only", "anneal"}:
        raise ValueError("dist_match_schedule must be 'warmup_only' or 'anneal'.")
    if cfg["dist_match_anneal_kind"] not in {"linear", "cosine"}:
        raise ValueError("dist_match_anneal_kind must be 'linear' or 'cosine'.")
    if cfg["dist_match_warmup_lr_mult"] < 0:
        raise ValueError("dist_match_warmup_lr_mult cannot be negative.")
    if cfg["dist_match_stage2_lr_mult_start"] < 0 or cfg["dist_match_stage2_lr_mult_end"] < 0:
        raise ValueError("dist_match stage-2 multipliers cannot be negative.")
