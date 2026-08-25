# VQ Experiment Study

This project refactors the notebook experiments into a reproducible CIFAR-10
ablation runner for VQ-VAE, ProVQ warmup, SimVQ, K-means initialization,
covariance initialization, codebook pre-adaptation, and distribution matching.

## Main commands

List experiments:

```bash
python -m vqexp.suite list --suite configs/suites/vq_ablation.json
```

Run one experiment locally:

```bash
python -m vqexp.train --suite configs/suites/vq_ablation.json --experiment vanilla --seed 42
```

Smoke test without downloading CIFAR-10:

```bash
python -m vqexp.train --suite configs/suites/smoke.json --experiment smoke_vanilla --seed 42
```

Aggregate results:

```bash
python -m vqexp.aggregate --root runs --suite-name vq_ablation_cifar10
```

Launch on Modal T4:

```bash
modal run modal_app.py --group core --seeds 42,3407,2026 --max-parallel 4
```

The Modal volume is mounted at `/vq`, so remote outputs default to
`/vq/runs/vq_ablation_cifar10`.
