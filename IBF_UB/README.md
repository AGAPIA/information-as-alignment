# IBF_UB Technical Review

This folder contains an independent technical review of the CIFAR-100 continual-learning experiment from [Information as Structural Alignment](https://arxiv.org/abs/2604.07108), plus two comparison scripts used for the aligned baseline checks.

## Files

- `IBF_UB_review.md`: review, ML translation, baseline comparison, use-case discussion, and ablation conclusions.
- `compare_cifar100_continual.py`: evaluates linear/MLP/deep-MLP classifier heads under fine-tuning and replay on the same frozen `ViT-B/16 + PCA` features.
- `compare_cifar100_frozen_features.py`: evaluates notebook-style MLP, replay, and EWC baselines on the frozen feature pipeline.

## Running

Run commands from the repository root so the scripts can find `CIFAR-paper-results.json`, `data/`, and `cache/` using their default paths.

```bash
python IBF_UB/compare_cifar100_frozen_features.py
python IBF_UB/compare_cifar100_continual.py
```

Both scripts can reuse the same frozen-feature cache:

```text
cache/cifar100_vitb16_pca64_features.npz
```

Use `--overwrite-cache` only when you intentionally want to recompute the frozen `ViT-B/16 + PCA` features.

## Scope

The review focuses on Domain III / CIFAR-100 because it is the closest implemented domain to a standard continual-learning benchmark with comparable baselines. RRW and Chess are discussed as supporting mechanism and strategic-decision domains, but the practical comparison is centered on CIFAR-100.
