# ProVQ Study

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

![[Pasted image 20260827023244.png]]
![[Pasted image 20260827023322.png]]