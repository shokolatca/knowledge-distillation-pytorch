#!/usr/bin/env python3
"""Evaluate MMS LID student checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from mms_lid.config import Params
from mms_lid.dataset import DistillSegmentDataset
from mms_lid.models.student_cnn import StudentCNN
from train_mms_lid import evaluate_clip_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MMS LID student")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--restore_file", default="best", help="best or last")
    parser.add_argument("--split", default=None, help="Split in manifest to evaluate")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    params = Params(model_dir / "params.json")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() and params.cuda else "cpu")

    dataset = DistillSegmentDataset(
        manifest_path=params.dev_manifest,
        split=args.split if args.split is not None else getattr(params, "dev_split", "dev"),
        sample_rate=params.sample_rate,
        window_sec=params.window_sec,
        hop_sec=params.hop_sec,
        n_mels=params.n_mels,
        n_fft=params.n_fft,
        hop_length=params.mel_hop_length,
        win_length=params.win_length,
        num_classes=params.num_classes,
        pseudo_labels_path=getattr(params, "dev_pseudo_labels", None),
        prefer_ground_truth=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = StudentCNN(
        num_classes=params.num_classes,
        base_channels=params.base_channels,
        dropout_rate=params.dropout_rate,
    ).to(device)

    checkpoint_path = model_dir / f"{args.restore_file}.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    metrics = evaluate_clip_level(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=params.num_classes,
    )

    logging.info("Eval metrics: %s", metrics)

    output_path = Path(args.output) if args.output else model_dir / f"metrics_{args.split}_{args.restore_file}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({k: float(v) for k, v in metrics.items()}, handle, indent=4)


if __name__ == "__main__":
    main()
