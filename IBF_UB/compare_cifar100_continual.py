#!/usr/bin/env python
"""
Split-CIFAR-100 baselines under controlled representation sharing.

Default mode evaluates linear and MLP heads on frozen ViT-B/16 + PCA features.
Optional raw mode evaluates CNN baselines on images. The script reports
Task-IL, Class-IL, backward transfer, and runtime.
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from compare_cifar100_frozen_features import load_or_compute_frozen_features


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


class IndexedDataset(Dataset):
    """Return (image, label, original_index) for easier bookkeeping."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, idx


class LinearHead(nn.Module):
    """Linear classifier on frozen 64D features."""

    def __init__(self, z_dim: int = 64, num_classes: int = N_CLASSES):
        super().__init__()
        self.classifier = nn.Linear(z_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPHead(nn.Module):
    """Small MLP head on frozen 64D features."""

    def __init__(self, z_dim: int = 64, hidden: int = 128, num_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLPHead(nn.Module):
    """Deeper MLP head on frozen 64D features."""

    def __init__(self, z_dim: int = 64, hidden: int = 256, num_classes: int = N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_resnet18_cifar(num_classes: int = N_CLASSES) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def make_model(name: str, num_classes: int = N_CLASSES, z_dim: int = 64) -> nn.Module:
    key = name.lower()
    if key == "linear":
        return LinearHead(z_dim=z_dim, num_classes=num_classes)
    if key == "mlp":
        return MLPHead(z_dim=z_dim, num_classes=num_classes)
    if key == "deepmlp":
        return DeepMLPHead(z_dim=z_dim, num_classes=num_classes)
    if key == "smallcnn":
        return SmallCNN(num_classes=num_classes)
    if key == "resnet18":
        return build_resnet18_cifar(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


class ReservoirReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.labels)

    def add_examples(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        for img, label in zip(images, labels):
            self.seen += 1
            img_cpu = img.detach().cpu().clone()
            label_int = int(label)
            if len(self.labels) < self.capacity:
                self.images.append(img_cpu)
                self.labels.append(label_int)
                continue
            j = random.randint(0, self.seen - 1)
            if j < self.capacity:
                self.images[j] = img_cpu
                self.labels[j] = label_int

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if not self.labels:
            return None
        take = min(batch_size, len(self.labels))
        idx = random.sample(range(len(self.labels)), take)
        x = torch.stack([self.images[i] for i in idx]).to(device)
        y = torch.tensor([self.labels[i] for i in idx], device=device)
        return x, y


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


def maybe_limit_indices(indices: List[int], max_per_class: int | None) -> List[int]:
    if max_per_class is None:
        return indices
    return indices[:max_per_class]


@dataclass
class ExperimentConfig:
    input_mode: str
    model: str
    strategy: str
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
    num_workers: int
    train_per_class: int | None
    test_per_class: int | None
    device: str


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    acc_taskil: List[List[float | None]]
    acc_classil: List[List[float | None]]
    avg_taskil: float
    avg_classil: float
    bt_taskil: float
    bt_classil: float
    elapsed_sec: float


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    loader: DataLoader,
    task_classes: Sequence[int],
    seen_classes: Sequence[int],
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    n = 0
    task_correct = 0.0
    class_correct = 0.0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)

        task_mask = torch.full((logits.size(1),), float("-inf"), device=device)
        for c in task_classes:
            task_mask[c] = 0.0
        task_preds = (logits + task_mask.unsqueeze(0)).argmax(dim=1)

        seen_mask = torch.full((logits.size(1),), float("-inf"), device=device)
        for c in seen_classes:
            seen_mask[c] = 0.0
        class_preds = (logits + seen_mask.unsqueeze(0)).argmax(dim=1)

        bs = labels.size(0)
        n += bs
        task_correct += (task_preds == labels).float().sum().item()
        class_correct += (class_preds == labels).float().sum().item()

    return task_correct / max(1, n), class_correct / max(1, n)


@torch.no_grad()
def evaluate_feature_task(
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
        task_preds = (logits + task_mask.unsqueeze(0)).argmax(dim=1)
        class_preds = (logits + seen_mask.unsqueeze(0)).argmax(dim=1)
        bs = y_b.size(0)
        total += bs
        task_correct += (task_preds == y_b).sum().item()
        class_correct += (class_preds == y_b).sum().item()

    return task_correct / max(1, total), class_correct / max(1, total)


def compute_bt(matrix: np.ndarray) -> float:
    vals = []
    num_tasks = matrix.shape[0]
    for task_id in range(num_tasks - 1):
        initial = matrix[task_id, task_id]
        final = matrix[task_id, num_tasks - 1]
        if not (math.isnan(initial) or math.isnan(final)):
            vals.append(final - initial)
    return float(np.mean(vals)) if vals else 0.0


def final_avg(matrix: np.ndarray) -> float:
    col = matrix[:, -1]
    vals = col[~np.isnan(col)]
    return float(np.mean(vals)) if len(vals) else 0.0


def matrix_to_lists(matrix: np.ndarray) -> List[List[float | None]]:
    out: List[List[float | None]] = []
    for row in matrix.tolist():
        out.append([None if isinstance(v, float) and math.isnan(v) else v for v in row])
    return out


def train_experiment(
    cfg: ExperimentConfig,
    tasks: List[List[int]],
    train_dataset: IndexedDataset,
    test_dataset: IndexedDataset,
    train_cls_indices: Dict[int, List[int]],
    test_cls_indices: Dict[int, List[int]],
) -> ExperimentResult:
    device = torch.device(cfg.device)
    model = make_model(cfg.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    buffer = ReservoirReplayBuffer(cfg.replay_buffer_size) if cfg.strategy == "replay" else None

    acc_taskil = np.full((cfg.num_tasks, cfg.num_tasks), np.nan, dtype=np.float32)
    acc_classil = np.full((cfg.num_tasks, cfg.num_tasks), np.nan, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()
    t0 = time.time()

    # Train the selected model sequentially over the task stream.
    for task_id, task_classes in enumerate(tasks):
        train_indices: List[int] = []
        for cls in task_classes:
            train_indices.extend(maybe_limit_indices(train_cls_indices[cls], cfg.train_per_class))
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )

        model.train()
        for _epoch in range(cfg.epochs_per_task):
            for images, labels, _ in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                if buffer is not None and len(buffer) > 0:
                    # Interleave replay samples with the current task batch.
                    replay_batch = buffer.sample(cfg.replay_batch_size, device)
                    if replay_batch is not None:
                        replay_x, replay_y = replay_batch
                        replay_logits = model(replay_x)
                        loss = loss + criterion(replay_logits, replay_y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if buffer is not None:
            # Add the completed task to the replay buffer.
            refill_loader = DataLoader(
                train_subset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
            )
            for images, labels, _ in refill_loader:
                buffer.add_examples(images, labels)

        seen_classes: List[int] = []
        for prev in range(task_id + 1):
            seen_classes.extend(tasks[prev])

        # Re-evaluate all tasks seen so far after each task update.
        for eval_task_id in range(task_id + 1):
            eval_classes = tasks[eval_task_id]
            test_indices: List[int] = []
            for cls in eval_classes:
                test_indices.extend(maybe_limit_indices(test_cls_indices[cls], cfg.test_per_class))
            test_subset = Subset(test_dataset, test_indices)
            test_loader = DataLoader(
                test_subset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
            )
            task_acc, class_acc = evaluate_task(
                model, test_loader, eval_classes, seen_classes, device
            )
            acc_taskil[eval_task_id, task_id] = task_acc
            acc_classil[eval_task_id, task_id] = class_acc

    elapsed_sec = time.time() - t0

    return ExperimentResult(
        config=cfg,
        acc_taskil=matrix_to_lists(acc_taskil),
        acc_classil=matrix_to_lists(acc_classil),
        avg_taskil=final_avg(acc_taskil),
        avg_classil=final_avg(acc_classil),
        bt_taskil=compute_bt(acc_taskil),
        bt_classil=compute_bt(acc_classil),
        elapsed_sec=elapsed_sec,
    )


def train_feature_experiment(
    cfg: ExperimentConfig,
    tasks: List[List[int]],
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    train_cls_indices: Dict[int, List[int]],
    test_cls_indices: Dict[int, List[int]],
) -> ExperimentResult:
    device = torch.device(cfg.device)
    model = make_model(cfg.model, num_classes=N_CLASSES, z_dim=cfg.z_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    buffer = ReservoirReplayBuffer(cfg.replay_buffer_size) if cfg.strategy == "replay" else None

    acc_taskil = np.full((cfg.num_tasks, cfg.num_tasks), np.nan, dtype=np.float32)
    acc_classil = np.full((cfg.num_tasks, cfg.num_tasks), np.nan, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()
    t0 = time.time()

    # Train the selected head on the shared frozen representation.
    for task_id, task_classes in enumerate(tasks):
        train_indices: List[int] = []
        for cls in task_classes:
            train_indices.extend(maybe_limit_indices(train_cls_indices[cls], cfg.train_per_class))
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

                if buffer is not None and len(buffer) > 0:
                    # Interleave replayed feature vectors with current-task data.
                    replay_batch = buffer.sample(cfg.replay_batch_size, device)
                    if replay_batch is not None:
                        replay_x, replay_y = replay_batch
                        loss = loss + criterion(model(replay_x), replay_y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if buffer is not None:
            # Add the completed task to the replay buffer.
            for start in range(0, nt, cfg.batch_size):
                z_b = torch.tensor(tz[start : start + cfg.batch_size], dtype=torch.float32)
                y_b = torch.tensor(ty[start : start + cfg.batch_size], dtype=torch.long)
                buffer.add_examples(z_b, y_b)

        seen_classes: List[int] = []
        for prev in range(task_id + 1):
            seen_classes.extend(tasks[prev])

        # Re-evaluate all tasks seen so far after each task update.
        for eval_task_id in range(task_id + 1):
            eval_classes = tasks[eval_task_id]
            test_indices: List[int] = []
            for cls in eval_classes:
                test_indices.extend(maybe_limit_indices(test_cls_indices[cls], cfg.test_per_class))
            task_acc, class_acc = evaluate_feature_task(
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

    elapsed_sec = time.time() - t0

    return ExperimentResult(
        config=cfg,
        acc_taskil=matrix_to_lists(acc_taskil),
        acc_classil=matrix_to_lists(acc_classil),
        avg_taskil=final_avg(acc_taskil),
        avg_classil=final_avg(acc_classil),
        bt_taskil=compute_bt(acc_taskil),
        bt_classil=compute_bt(acc_classil),
        elapsed_sec=elapsed_sec,
    )


def load_ibf_reference(path: Path) -> Dict[str, object] | None:
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
            "n_value_centers": data["runs"]["full_42"]["n_value_centers"],
            "n_agency_centers": data["runs"]["full_42"]["n_agency_centers"],
        }
    }
    if "avg_replay" in data["runs"]["full_42"]:
        ref["notebook_replay_mlp"] = {
            "avg_taskil": data["runs"]["full_42"]["avg_replay"],
            "bt_taskil": data["runs"]["full_42"]["BT_replay"],
            "classil": data.get("classil", {}).get("replay"),
        }
    return ref


def print_summary(results: List[ExperimentResult], ibf_ref: Dict[str, object] | None) -> None:
    rows = []
    for r in results:
        rows.append(
            (
                f"{r.config.model}/{r.config.strategy}",
                f"{r.avg_taskil:.4f}",
                f"{r.avg_classil:.4f}",
                f"{r.bt_taskil:+.4f}",
                f"{r.elapsed_sec / 60.0:.1f}m",
            )
        )

    print("\nResults")
    print("-" * 78)
    print(f"{'experiment':28} {'task-il':>10} {'class-il':>10} {'bt':>10} {'time':>10}")
    print("-" * 78)
    for row in rows:
        print(f"{row[0]:28} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
    print("-" * 78)

    if ibf_ref is not None:
        ibf = ibf_ref["ibf_full_42"]
        print("Reference from CIFAR-paper-results.json")
        print(
            "  IBF full_42:"
            f" task-il(linear)={ibf['avg_taskil_lin']:.4f},"
            f" class-il(linear)={ibf['avg_classil_lin']:.4f},"
            f" bt(linear)={ibf['bt_taskil_lin']:+.4f},"
            f" task-il(log)={ibf['avg_taskil_log']:.4f},"
            f" bt(log)={ibf['bt_taskil_log']:+.4f},"
            f" time={ibf['elapsed_hours']:.2f}h"
        )
        replay = ibf_ref.get("notebook_replay_mlp")
        if replay is not None:
            print(
                "  Notebook replay MLP:"
                f" task-il={replay['avg_taskil']:.4f},"
                f" class-il={replay['classil']:.4f},"
                f" bt={replay['bt_taskil']:+.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark standard Split-CIFAR-100 continual-learning baselines. "
            "By default this uses the same frozen ViT-B/16 + PCA features as "
            "the notebook. Use --input-mode raw for raw-image CNN baselines."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("cache/cifar100_vitb16_pca64_features.npz"),
        help="Frozen-feature cache used when --input-mode feature.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute the frozen ViT-B/16 + PCA feature cache.",
    )
    parser.add_argument("--out", type=Path, default=Path("cifar_compare_results.json"))
    parser.add_argument(
        "--ibf-reference",
        type=Path,
        default=Path("CIFAR-paper-results.json"),
        help="Optional notebook reference JSON for side-by-side reporting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_NUM_TASKS)
    parser.add_argument("--classes-per-task", type=int, default=DEFAULT_CLASSES_PER_TASK)
    parser.add_argument(
        "--input-mode",
        type=str,
        default="feature",
        choices=["feature", "raw"],
        help="feature = frozen ViT-B/16 + PCA inputs, raw = raw-image CNN inputs.",
    )
    parser.add_argument("--z-dim", type=int, default=DEFAULT_Z_DIM)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "mlp", "deepmlp"],
        choices=["linear", "mlp", "deepmlp", "smallcnn", "resnet18"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["finetune", "replay"],
        choices=["finetune", "replay"],
    )
    parser.add_argument(
        "--epochs-per-task",
        type=int,
        default=10,
        help="Use 50 to get closer to the notebook's task schedule.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--replay-buffer-size", type=int, default=5000)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=None,
        help="Optional cap for quick experiments or smoke tests.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=None,
        help="Optional cap for quick experiments or smoke tests.",
    )
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
        raise ValueError("num_tasks * classes_per_task cannot exceed 100 for CIFAR-100")

    feature_models = {"linear", "mlp", "deepmlp"}
    raw_models = {"smallcnn", "resnet18"}
    chosen_models = set(args.models)
    if args.input_mode == "feature" and not chosen_models.issubset(feature_models):
        raise ValueError(
            "Feature mode only supports models: linear, mlp, deepmlp. "
            "Use --input-mode raw for smallcnn/resnet18."
        )
    if args.input_mode == "raw" and not chosen_models.issubset(raw_models):
        raise ValueError(
            "Raw mode only supports models: smallcnn, resnet18. "
            "Use --input-mode feature for linear/mlp/deepmlp."
        )

    # Raw-image transforms are used only when --input-mode raw.
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_base = datasets.CIFAR100(
        root=str(args.data_dir), train=True, download=True, transform=train_tf
    )
    test_base = datasets.CIFAR100(
        root=str(args.data_dir), train=False, download=True, transform=test_tf
    )
    train_dataset = IndexedDataset(train_base)
    test_dataset = IndexedDataset(test_base)

    # Fix the class order once and reuse it across all runs.
    tasks = build_task_splits(args.seed, args.num_tasks, args.classes_per_task)

    print("Task split:")
    for task_id, task_classes in enumerate(tasks):
        print(f"  Task {task_id:02d}: {task_classes}")

    results: List[ExperimentResult] = []
    if args.input_mode == "feature":
        # Load or compute the shared frozen ViT+PCA representation.
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

        for model_name in args.models:
            for strategy_name in args.strategies:
                cfg = ExperimentConfig(
                    input_mode=args.input_mode,
                    model=model_name,
                    strategy=strategy_name,
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
                    num_workers=args.num_workers,
                    train_per_class=args.train_per_class,
                    test_per_class=args.test_per_class,
                    device=args.device,
                )
                print(
                    f"\nRunning {cfg.model}/{cfg.strategy}"
                    f" on {cfg.device} for {cfg.epochs_per_task} epochs/task"
                    f" with frozen ViT-B/16 + PCA features"
                )
                result = train_feature_experiment(
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
    else:
        # Train raw-image baselines directly on CIFAR-100 pixels.
        train_cls_indices = indices_by_class(train_base.targets)
        test_cls_indices = indices_by_class(test_base.targets)
        for model_name in args.models:
            for strategy_name in args.strategies:
                cfg = ExperimentConfig(
                    input_mode=args.input_mode,
                    model=model_name,
                    strategy=strategy_name,
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
                    num_workers=args.num_workers,
                    train_per_class=args.train_per_class,
                    test_per_class=args.test_per_class,
                    device=args.device,
                )
                print(
                    f"\nRunning {cfg.model}/{cfg.strategy}"
                    f" on {cfg.device} for {cfg.epochs_per_task} epochs/task"
                    f" with raw CIFAR images"
                )
                result = train_experiment(
                    cfg=cfg,
                    tasks=tasks,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    train_cls_indices=train_cls_indices,
                    test_cls_indices=test_cls_indices,
                )
                results.append(result)

    ibf_ref = load_ibf_reference(args.ibf_reference)
    print_summary(results, ibf_ref)

    payload = {
        "config": {
            "seed": args.seed,
            "num_tasks": args.num_tasks,
            "classes_per_task": args.classes_per_task,
            "input_mode": args.input_mode,
            "z_dim": args.z_dim,
            "epochs_per_task": args.epochs_per_task,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "replay_buffer_size": args.replay_buffer_size,
            "replay_batch_size": args.replay_batch_size,
            "train_per_class": args.train_per_class,
            "test_per_class": args.test_per_class,
            "device": args.device,
            "feature_cache": str(args.feature_cache),
        },
        "tasks": tasks,
        "results": [
            {
                "config": asdict(r.config),
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
        "ibf_reference": ibf_ref,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
