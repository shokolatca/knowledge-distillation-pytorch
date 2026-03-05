#!/usr/bin/env python3
"""Train MMS LID student with offline pseudo labels."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from mms_lid.config import Params
from mms_lid.dataset import DistillSegmentDataset
from mms_lid.losses import kd_loss
from mms_lid.metrics import classification_report
from mms_lid.models.student_cnn import StudentCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MMS LID student model")
    parser.add_argument("--model_dir", required=True, help="Directory containing params.json")
    parser.add_argument(
        "--restore_file",
        default=None,
        help="Optional checkpoint name (best/last) to restore from",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(data: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({k: float(v) for k, v in data.items()}, handle, indent=4)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optim_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_dict"])
    return int(checkpoint.get("epoch", 0))


def build_dataloader(
    manifest_path: str,
    split: Optional[str],
    params: Params,
    pseudo_labels_path: Optional[str],
    shuffle: bool,
) -> DataLoader:
    dataset = DistillSegmentDataset(
        manifest_path=manifest_path,
        split=split,
        sample_rate=params.sample_rate,
        window_sec=params.window_sec,
        hop_sec=params.hop_sec,
        n_mels=params.n_mels,
        n_fft=params.n_fft,
        hop_length=params.mel_hop_length,
        win_length=params.win_length,
        num_classes=params.num_classes,
        pseudo_labels_path=pseudo_labels_path,
        prefer_ground_truth=params.prefer_ground_truth,
    )
    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=shuffle,
        num_workers=params.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def evaluate_clip_level(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()

    clip_logits_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
    clip_counts: Dict[str, int] = defaultdict(int)
    clip_targets: Dict[str, int] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            features = batch["features"].to(device, non_blocking=True)
            hard_target = batch["hard_target"].cpu().numpy()
            clip_ids = batch["clip_id"]

            logits = model(features).cpu().numpy()
            for idx, clip_id in enumerate(clip_ids):
                clip_logits_sum[clip_id] += logits[idx]
                clip_counts[clip_id] += 1
                clip_targets.setdefault(clip_id, int(hard_target[idx]))

    ordered_clip_ids = sorted(clip_logits_sum.keys())
    predictions = []
    targets = []

    for clip_id in ordered_clip_ids:
        mean_logits = clip_logits_sum[clip_id] / max(clip_counts[clip_id], 1)
        predictions.append(int(np.argmax(mean_logits)))
        targets.append(int(clip_targets[clip_id]))

    predictions_np = np.array(predictions, dtype=np.int64)
    targets_np = np.array(targets, dtype=np.int64)
    return classification_report(predictions_np, targets_np, num_classes=num_classes)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        hard_target = batch["hard_target"].to(device, non_blocking=True)
        teacher_logits = batch["teacher_logits"].to(device, non_blocking=True)
        teacher_mask = batch["has_teacher"].to(device, non_blocking=True).bool()

        optimizer.zero_grad(set_to_none=True)
        student_logits = model(features)
        loss, ce_part, kl_part = kd_loss(
            student_logits=student_logits,
            hard_targets=hard_target,
            teacher_logits=teacher_logits,
            alpha=alpha,
            temperature=temperature,
            teacher_mask=teacher_mask,
        )
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_part.item()) * batch_size
        total_kl += float(kl_part.item()) * batch_size
        count += batch_size

    if count == 0:
        return {"loss": 0.0, "ce": 0.0, "kl": 0.0}

    return {
        "loss": total_loss / count,
        "ce": total_ce / count,
        "kl": total_kl / count,
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    params = Params(model_dir / "params.json")
    set_seed(int(params.seed))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(model_dir / "train_mms_lid.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() and params.cuda else "cpu")
    logging.info("Using device: %s", device)

    train_dl = build_dataloader(
        manifest_path=params.train_manifest,
        split=getattr(params, "train_split", "train"),
        params=params,
        pseudo_labels_path=getattr(params, "train_pseudo_labels", None),
        shuffle=True,
    )
    dev_dl = build_dataloader(
        manifest_path=params.dev_manifest,
        split=getattr(params, "dev_split", "dev"),
        params=params,
        pseudo_labels_path=getattr(params, "dev_pseudo_labels", None),
        shuffle=False,
    )

    model = StudentCNN(
        num_classes=params.num_classes,
        base_channels=params.base_channels,
        dropout_rate=params.dropout_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(params.num_epochs), 1))

    start_epoch = 0
    if args.restore_file:
        checkpoint_path = model_dir / f"{args.restore_file}.pth.tar"
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        logging.info("Restored from %s at epoch %d", checkpoint_path, start_epoch)

    best_macro_f1 = -1.0

    for epoch in range(start_epoch, int(params.num_epochs)):
        logging.info("Epoch %d/%d", epoch + 1, int(params.num_epochs))
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            device=device,
            alpha=float(params.alpha),
            temperature=float(params.temperature),
        )
        scheduler.step()

        val_metrics = evaluate_clip_level(
            model=model,
            dataloader=dev_dl,
            device=device,
            num_classes=int(params.num_classes),
        )

        logging.info(
            "train loss=%.4f ce=%.4f kl=%.4f | dev acc=%.4f macro_f1=%.4f",
            train_metrics["loss"],
            train_metrics["ce"],
            train_metrics["kl"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )

        save_checkpoint(model, optimizer, epoch + 1, model_dir / "last.pth.tar")
        save_json(val_metrics, model_dir / "metrics_val_last_weights.json")

        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, optimizer, epoch + 1, model_dir / "best.pth.tar")
            save_json(val_metrics, model_dir / "metrics_val_best_weights.json")
            logging.info("Found new best macro_f1: %.4f", best_macro_f1)


if __name__ == "__main__":
    main()
