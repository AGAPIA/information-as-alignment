#!/usr/bin/env python
"""
Split-CIFAR-100 baselines on frozen ViT-B/16 + PCA features.

The script evaluates MLP, replay, and EWC baselines under the same
feature pipeline as Domain III and reports Task-IL, Class-IL,
backward transfer, and runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torchvision import datasets, models, transforms


N_CLASSES = 100
DEFAULT_NUM_TASKS = 20
DEFAULT_CLASSES_PER_TASK = 5
DEFAULT_Z_DIM = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"


def build_task_splits(
    seed: int,
    num_tasks: int,
    classes_per_task: int,
    num_classes: int = N_CLASSES,
) -> List[List[int]]:
    order = list(range(num_classes))
    rng = random.Random(seed)
    rng.shuffle(order)
    return [
        order[t * classes_per_task : (t + 1) * classes_per_task]
        for t in range(num_tasks)
    ]


def indices_by_class(targets: Sequence[int], num_classes: int = N_CLASSES) -> Dict[int, List[int]]:
    out = {c: [] for c in range(num_classes)}
    for idx, target in enumerate(targets):
        out[int(target)].append(idx)
    return out


def maybe_cap(indices: List[int], cap: int | None) -> List[int]:
    if cap is None:
        return indices
    return indices[:cap]


class ViTFeatureExtractor(nn.Module):
    """Extract CLS token embeddings (768D) from ViT-B/16."""

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.vit = vit_model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(x)
        n = x.shape[0]
        cls_tok = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]


class MLPClassifier(nn.Module):
    """Plain baseline on frozen 64D features."""

    def __init__(self, z_dim: int = DEFAULT_Z_DIM, hidden: int = 128, n_cls: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ReplayClassifier(MLPClassifier):
    def __init__(
        self,
        z_dim: int = DEFAULT_Z_DIM,
        hidden: int = 128,
        n_cls: int = N_CLASSES,
        buffer_size: int = 5000,
    ):
        super().__init__(z_dim=z_dim, hidden=hidden, n_cls=n_cls)
        self.buf_z: List[np.ndarray] = []
        self.buf_y: List[int] = []
        self.buf_size = buffer_size
        self.seen = 0

    def add_to_buffer(self, z_np: np.ndarray, y_np: np.ndarray) -> None:
        for i in range(len(z_np)):
            self.seen += 1
            if len(self.buf_z) < self.buf_size:
                self.buf_z.append(z_np[i].copy())
                self.buf_y.append(int(y_np[i]))
            else:
                # Reservoir sampling keeps a uniform sample over the stream.
                j = random.randint(0, self.seen - 1)
                if j < self.buf_size:
                    self.buf_z[j] = z_np[i].copy()
                    self.buf_y[j] = int(y_np[i])

    def get_replay_batch(self, batch_size: int = 64) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if not self.buf_z:
            return None, None
        idx = np.random.choice(len(self.buf_z), min(batch_size, len(self.buf_z)), replace=False)
        z = np.array([self.buf_z[i] for i in idx], dtype=np.float32)
        y = np.array([self.buf_y[i] for i in idx], dtype=np.int64)
        return z, y


class EWCClassifier(MLPClassifier):
    def __init__(
        self,
        z_dim: int = DEFAULT_Z_DIM,
        hidden: int = 128,
        n_cls: int = N_CLASSES,
        ewc_lambda: float = 1000.0,
    ):
        super().__init__(z_dim=z_dim, hidden=hidden, n_cls=n_cls)
        self.ewc_lambda = ewc_lambda
        self.fisher: Dict[str, torch.Tensor] = {}
        self.params_old: Dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        z_np: np.ndarray,
        y_np: np.ndarray,
        device: torch.device,
        n_samples: int = 500,
    ) -> None:
        self.fisher = {}
        self.params_old = {}
        for name, p in self.named_parameters():
            self.fisher[name] = torch.zeros_like(p)
            self.params_old[name] = p.data.clone()

        self.train()
        n = min(n_samples, len(z_np))
        idx = np.random.choice(len(z_np), n, replace=False)
        for start in range(0, n, 64):
            batch_idx = idx[start : start + 64]
            z_b = torch.tensor(z_np[batch_idx], dtype=torch.float32, device=device)
            y_b = torch.tensor(y_np[batch_idx], dtype=torch.long, device=device)
            logits = self(z_b)
            loss = nn.CrossEntropyLoss()(logits, y_b)
            self.zero_grad()
            loss.backward()
            for name, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher[name] += p.grad.data ** 2
        n_batches = max(1, (n + 63) // 64)
        for name in self.fisher:
            self.fisher[name] /= n_batches

    def ewc_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, p in self.named_parameters():
            if name in self.fisher:
                penalty = penalty + (self.fisher[name] * (p - self.params_old[name]) ** 2).sum()
        return self.ewc_lambda * penalty


@dataclass
class BaselineConfig:
    seed: int
    num_tasks: int
    classes_per_task: int
    z_dim: int
    epochs_per_task: int
    batch_size: int
    lr: float
    weight_decay: float
    replay_buffer_size: int
    replay_batch_size: int
    ewc_lambda: float
    fisher_samples: int
    train_per_class: int | None
    test_per_class: int | None
    device: str


@dataclass
class BaselineResult:
    name: str
    acc_taskil: List[List[float | None]]
    acc_classil: List[List[float | None]]
    avg_taskil: float
    avg_classil: float
    bt_taskil: float
    bt_classil: float
    elapsed_sec: float


def matrix_to_lists(matrix: np.ndarray) -> List[List[float | None]]:
    out: List[List[float | None]] = []
    for row in matrix.tolist():
        out.append([None if isinstance(v, float) and math.isnan(v) else v for v in row])
    return out


def compute_bt(matrix: np.ndarray) -> float:
    vals = []
    num_tasks = matrix.shape[0]
    for t in range(num_tasks - 1):
        initial = matrix[t, t]
        final = matrix[t, num_tasks - 1]
        if not (math.isnan(initial) or math.isnan(final)):
            vals.append(final - initial)
    return float(np.mean(vals)) if vals else 0.0


def final_avg(matrix: np.ndarray) -> float:
    col = matrix[:, -1]
    vals = col[~np.isnan(col)]
    return float(np.mean(vals)) if len(vals) else 0.0


def encode_dataset_to_768(
    dataset: datasets.CIFAR100,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_feats = []
    with torch.no_grad():
        for images, _labels in loader:
            feats = encoder(images.to(device)).cpu().numpy()
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def load_or_compute_frozen_features(
    data_dir: Path,
    cache_path: Path,
    z_dim: int,
    device: torch.device,
    num_workers: int,
    overwrite_cache: bool = False,
) -> Dict[str, np.ndarray]:
    if cache_path.exists() and not overwrite_cache:
        return dict(np.load(cache_path, allow_pickle=False))

    # Use a fixed feature pipeline shared by all downstream baselines.
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    cifar_train = datasets.CIFAR100(
        root=str(data_dir), train=True, download=True, transform=train_tf
    )
    cifar_test = datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=test_tf
    )

    vit = models.vit_b_16(weights="IMAGENET1K_V1").to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False
    encoder = ViTFeatureExtractor(vit).to(device)

    # Fit PCA once and apply the same projection to train and test sets.
    print("Encoding test set to 768D for PCA fit...")
    test_768 = encode_dataset_to_768(cifar_test, encoder, device, num_workers=num_workers)

    print(f"Fitting PCA: 768D -> {z_dim}D")
    pca = PCA(n_components=z_dim)
    pca.fit(test_768)
    pca_mean = pca.mean_.astype(np.float32)
    pca_components = pca.components_.astype(np.float32)

    z64_test = ((test_768 - pca_mean) @ pca_components.T).astype(np.float32)

    print("Encoding training set to 64D...")
    train_768 = encode_dataset_to_768(cifar_train, encoder, device, num_workers=num_workers)
    z64_train = ((train_768 - pca_mean) @ pca_components.T).astype(np.float32)

    payload = {
        "z_train": z64_train,
        "z_test": z64_test,
        "train_labels": np.array(cifar_train.targets, dtype=np.int64),
        "test_labels": np.array(cifar_test.targets, dtype=np.int64),
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


@torch.no_grad()
def evaluate_features(
    model: nn.Module,
    z_test: np.ndarray,
    y_test: np.ndarray,
    task_indices: List[int],
    task_classes: Sequence[int],
    seen_classes: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    z = z_test[np.array(task_indices)]
    y = y_test[np.array(task_indices)]
    task_correct = 0
    class_correct = 0
    total = 0

    # Restrict predictions to task-local or seen-class candidate sets.
    task_mask = torch.full((N_CLASSES,), float("-inf"), device=device)
    for c in task_classes:
        task_mask[c] = 0.0

    seen_mask = torch.full((N_CLASSES,), float("-inf"), device=device)
    for c in seen_classes:
        seen_mask[c] = 0.0

    for start in range(0, len(z), batch_size):
        z_b = torch.tensor(z[start : start + batch_size], dtype=torch.float32, device=device)
        y_b = torch.tensor(y[start : start + batch_size], dtype=torch.long, device=device)
        logits = model(z_b)
        pred_task = (logits + task_mask.unsqueeze(0)).argmax(dim=1)
        pred_class = (logits + seen_mask.unsqueeze(0)).argmax(dim=1)
        bs = y_b.size(0)
        total += bs
        task_correct += (pred_task == y_b).sum().item()
        class_correct += (pred_class == y_b).sum().item()

    return task_correct / max(1, total), class_correct / max(1, total)


def train_one_baseline(
    name: str,
    cfg: BaselineConfig,
    tasks: List[List[int]],
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    train_cls_indices: Dict[int, List[int]],
    test_cls_indices: Dict[int, List[int]],
) -> BaselineResult:
    device = torch.device(cfg.device)

    # Select the baseline family under a shared feature representation.
    if name == "mlp":
        model: nn.Module = MLPClassifier(cfg.z_dim, 128, N_CLASSES).to(device)
    elif name == "replay":
        model = ReplayClassifier(cfg.z_dim, 128, N_CLASSES, cfg.replay_buffer_size).to(device)
    elif name == "ewc":
        model = EWCClassifier(cfg.z_dim, 128, N_CLASSES, cfg.ewc_lambda).to(device)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    acc_taskil = np.full((cfg.num_tasks, cfg.num_tasks), np.nan, dtype=np.float32)
    acc_classil = np.full((cfg.num_tasks, cfg.num_tasks), np.nan, dtype=np.float32)

    t0 = time.time()

    # Train the selected baseline sequentially over the task stream.
    for task_id in range(cfg.num_tasks):
        task_classes = tasks[task_id]
        train_indices: List[int] = []
        for cls in task_classes:
            train_indices.extend(maybe_cap(train_cls_indices[cls], cfg.train_per_class))
        train_indices = np.array(train_indices, dtype=np.int64)
        tz = z_train[train_indices]
        ty = y_train[train_indices]
        nt = len(tz)

        for _epoch in range(cfg.epochs_per_task):
            perm = np.random.permutation(nt)
            for start in range(0, nt, cfg.batch_size):
                bidx = perm[start : start + cfg.batch_size]
                z_b = torch.tensor(tz[bidx], dtype=torch.float32, device=device)
                y_b = torch.tensor(ty[bidx], dtype=torch.long, device=device)

                model.train()
                loss = criterion(model(z_b), y_b)

                if name == "replay":
                    # Interleave replayed feature vectors with current-task data.
                    replay_model = model  # type: ignore[assignment]
                    rz, ry = replay_model.get_replay_batch(cfg.replay_batch_size)
                    if rz is not None:
                        rz_t = torch.tensor(rz, dtype=torch.float32, device=device)
                        ry_t = torch.tensor(ry, dtype=torch.long, device=device)
                        loss = loss + criterion(replay_model(rz_t), ry_t)

                if name == "ewc":
                    # Penalize drift from parameters important to earlier tasks.
                    ewc_model = model  # type: ignore[assignment]
                    loss = loss + ewc_model.ewc_penalty()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if name == "replay":
            # Add the completed task to the replay buffer.
            replay_model = model  # type: ignore[assignment]
            replay_model.add_to_buffer(tz, ty)

        if name == "ewc":
            # Estimate parameter importance at the end of each task.
            ewc_model = model  # type: ignore[assignment]
            ewc_model.compute_fisher(tz, ty, device=device, n_samples=cfg.fisher_samples)

        seen_classes: List[int] = []
        for t in range(task_id + 1):
            seen_classes.extend(tasks[t])

        for eval_task_id in range(task_id + 1):
            eval_classes = tasks[eval_task_id]
            test_indices: List[int] = []
            for cls in eval_classes:
                test_indices.extend(maybe_cap(test_cls_indices[cls], cfg.test_per_class))
            # Re-evaluate all tasks seen so far after each task update.
            task_acc, class_acc = evaluate_features(
                model=model,
                z_test=z_test,
                y_test=y_test,
                task_indices=test_indices,
                task_classes=eval_classes,
                seen_classes=seen_classes,
                batch_size=cfg.batch_size,
                device=device,
            )
            acc_taskil[eval_task_id, task_id] = task_acc
            acc_classil[eval_task_id, task_id] = class_acc

    return BaselineResult(
        name=name,
        acc_taskil=matrix_to_lists(acc_taskil),
        acc_classil=matrix_to_lists(acc_classil),
        avg_taskil=final_avg(acc_taskil),
        avg_classil=final_avg(acc_classil),
        bt_taskil=compute_bt(acc_taskil),
        bt_classil=compute_bt(acc_classil),
        elapsed_sec=time.time() - t0,
    )


def load_reference(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    ref = {
        "ibf_full_42": {
            "avg_taskil_lin": data["runs"]["full_42"]["avg_taskil_lin"],
            "avg_classil_lin": data["runs"]["full_42"]["avg_classil_lin"],
            "avg_taskil_log": data["runs"]["full_42"]["avg_taskil_log"],
            "avg_classil_log": data["runs"]["full_42"]["avg_classil_log"],
            "bt_taskil_lin": data["runs"]["full_42"]["BT_taskil_lin"],
            "bt_taskil_log": data["runs"]["full_42"]["BT_taskil_log"],
            "elapsed_hours": data["runs"]["full_42"]["elapsed"] / 3600.0,
        },
        "notebook_baselines": {
            "mlp_taskil": data["runs"]["full_42"].get("avg_mlp"),
            "replay_taskil": data["runs"]["full_42"].get("avg_replay"),
            "ewc_taskil": data["runs"]["full_42"].get("avg_ewc"),
            "mlp_bt": data["runs"]["full_42"].get("BT_mlp"),
            "replay_bt": data["runs"]["full_42"].get("BT_replay"),
            "ewc_bt": data["runs"]["full_42"].get("BT_ewc"),
            "mlp_classil": data.get("classil", {}).get("mlp"),
            "replay_classil": data.get("classil", {}).get("replay"),
            "ewc_classil": data.get("classil", {}).get("ewc"),
        },
    }
    return ref


def print_summary(results: List[BaselineResult], ref: Dict[str, object] | None) -> None:
    print("\nResults")
    print("-" * 74)
    print(f"{'baseline':16} {'task-il':>10} {'class-il':>10} {'bt':>10} {'time':>10}")
    print("-" * 74)
    for r in results:
        print(
            f"{r.name:16} {r.avg_taskil:>10.4f} {r.avg_classil:>10.4f}"
            f" {r.bt_taskil:>+10.4f} {r.elapsed_sec / 60.0:>9.1f}m"
        )
    print("-" * 74)

    if ref is not None:
        ibf = ref["ibf_full_42"]
        print(
            "Notebook IBF full_42:"
            f" task-il(linear)={ibf['avg_taskil_lin']:.4f},"
            f" class-il(linear)={ibf['avg_classil_lin']:.4f},"
            f" bt(linear)={ibf['bt_taskil_lin']:+.4f},"
            f" task-il(log)={ibf['avg_taskil_log']:.4f},"
            f" bt(log)={ibf['bt_taskil_log']:+.4f},"
            f" time={ibf['elapsed_hours']:.2f}h"
        )
        base = ref["notebook_baselines"]
        print(
            "Notebook frozen-feature baselines:"
            f" mlp={base['mlp_taskil']:.4f},"
            f" replay={base['replay_taskil']:.4f},"
            f" ewc={base['ewc_taskil']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split-CIFAR-100 baselines on frozen ViT+PCA features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("cache/cifar100_vitb16_pca64_features.npz"),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("cifar_frozen_feature_compare.json"))
    parser.add_argument("--reference", type=Path, default=Path("CIFAR-paper-results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_NUM_TASKS)
    parser.add_argument("--classes-per-task", type=int, default=DEFAULT_CLASSES_PER_TASK)
    parser.add_argument("--z-dim", type=int, default=DEFAULT_Z_DIM)
    parser.add_argument("--epochs-per-task", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["mlp", "replay", "ewc"],
        choices=["mlp", "replay", "ewc"],
    )
    parser.add_argument("--replay-buffer-size", type=int, default=5000)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--ewc-lambda", type=float, default=1000.0)
    parser.add_argument("--fisher-samples", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Using device: {describe_device(device)}")

    if args.num_tasks * args.classes_per_task > N_CLASSES:
        raise ValueError("num_tasks * classes_per_task cannot exceed 100")

    # Load or compute the shared frozen representation.
    features = load_or_compute_frozen_features(
        data_dir=args.data_dir,
        cache_path=args.feature_cache,
        z_dim=args.z_dim,
        device=device,
        num_workers=args.num_workers,
        overwrite_cache=args.overwrite_cache,
    )
    z_train = features["z_train"]
    z_test = features["z_test"]
    y_train = features["train_labels"]
    y_test = features["test_labels"]

    train_cls_indices = indices_by_class(y_train)
    test_cls_indices = indices_by_class(y_test)

    # Fix the class order once and reuse it across all runs.
    tasks = build_task_splits(args.seed, args.num_tasks, args.classes_per_task)

    print("Task split:")
    for task_id, task_classes in enumerate(tasks):
        print(f"  Task {task_id:02d}: {task_classes}")

    cfg = BaselineConfig(
        seed=args.seed,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        z_dim=args.z_dim,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        replay_buffer_size=args.replay_buffer_size,
        replay_batch_size=args.replay_batch_size,
        ewc_lambda=args.ewc_lambda,
        fisher_samples=args.fisher_samples,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        device=args.device,
    )

    results = []
    for baseline in args.baselines:
        print(f"\nRunning baseline: {baseline}")
        result = train_one_baseline(
            name=baseline,
            cfg=cfg,
            tasks=tasks,
            z_train=z_train,
            y_train=y_train,
            z_test=z_test,
            y_test=y_test,
            train_cls_indices=train_cls_indices,
            test_cls_indices=test_cls_indices,
        )
        results.append(result)

    ref = load_reference(args.reference)
    print_summary(results, ref)

    payload = {
        "config": asdict(cfg),
        "tasks": tasks,
        "feature_cache": str(args.feature_cache),
        "results": [
            {
                "name": r.name,
                "acc_taskil": r.acc_taskil,
                "acc_classil": r.acc_classil,
                "avg_taskil": r.avg_taskil,
                "avg_classil": r.avg_classil,
                "bt_taskil": r.bt_taskil,
                "bt_classil": r.bt_classil,
                "elapsed_sec": r.elapsed_sec,
            }
            for r in results
        ],
        "reference": ref,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
