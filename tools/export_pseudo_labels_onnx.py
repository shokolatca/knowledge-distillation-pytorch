#!/usr/bin/env python3
"""Export offline pseudo labels from an ONNX teacher model."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from tqdm import tqdm

from mms_lid.features import extract_log_mel
from mms_lid.manifest import load_manifest
from mms_lid.pseudo_labels import ClipPseudoLabel, SegmentPseudoLabel, save_pseudo_labels_npz
from mms_lid.segmentation import compute_segment_starts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pseudo labels from ONNX model")
    parser.add_argument("--manifest", required=True, help="Path to train/dev manifest (.jsonl/.csv)")
    parser.add_argument("--onnx", required=True, help="Path to teacher ONNX model")
    parser.add_argument("--output", required=True, help="Output .npz pseudo-label path")
    parser.add_argument("--split", default="train", help="Split to export")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--input-format", choices=["waveform", "logmel"], default="waveform")
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--mel-hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--transpose-logmel", action="store_true")
    return parser


def _get_resampler(cache: Dict[int, torchaudio.transforms.Resample], src_sr: int, dst_sr: int):
    if src_sr not in cache:
        cache[src_sr] = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
    return cache[src_sr]


def _prepare_input(
    segment_waveform: torch.Tensor,
    input_rank: int,
    input_format: str,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    mel_hop_length: int,
    win_length: int,
    transpose_logmel: bool,
) -> np.ndarray:
    if input_format == "waveform":
        waveform = segment_waveform.squeeze(0).cpu().numpy().astype(np.float32)
        if input_rank == 1:
            return waveform
        if input_rank == 2:
            return waveform[None, :]
        if input_rank == 3:
            return waveform[None, None, :]
        raise ValueError(f"Unsupported ONNX input rank {input_rank} for waveform input")

    feature = extract_log_mel(
        waveform=segment_waveform,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=mel_hop_length,
        win_length=win_length,
    ).cpu().numpy().astype(np.float32)  # [1, n_mels, frames]

    if transpose_logmel:
        feature = np.transpose(feature, (0, 2, 1))

    if input_rank == 2:
        return feature.reshape(1, -1)
    if input_rank == 3:
        return feature
    if input_rank == 4:
        return feature[None, :, :, :]
    raise ValueError(f"Unsupported ONNX input rank {input_rank} for logmel input")


def _extract_logits(output_tensor: np.ndarray) -> np.ndarray:
    logits = np.asarray(output_tensor)
    logits = np.squeeze(logits)
    if logits.ndim == 1:
        return logits.astype(np.float32)
    if logits.ndim == 2 and logits.shape[0] == 1:
        return logits[0].astype(np.float32)
    raise ValueError(f"Expected model output of shape [C] or [1, C], got {logits.shape}")


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest_entries = load_manifest(args.manifest, split=args.split)
    providers = [args.provider]
    session = ort.InferenceSession(args.onnx, providers=providers)

    input_meta = (
        next(x for x in session.get_inputs() if x.name == args.input_name)
        if args.input_name
        else session.get_inputs()[0]
    )
    output_meta = (
        next(x for x in session.get_outputs() if x.name == args.output_name)
        if args.output_name
        else session.get_outputs()[0]
    )

    input_name = input_meta.name
    output_name = output_meta.name
    input_rank = len(input_meta.shape)

    logging.info(
        "Loaded ONNX model. input=%s rank=%s output=%s provider=%s",
        input_name,
        input_rank,
        output_name,
        providers,
    )

    sample_rate = int(args.sample_rate)
    window_samples = int(round(args.window_sec * sample_rate))
    hop_samples = int(round(args.hop_sec * sample_rate))

    segment_labels: List[SegmentPseudoLabel] = []
    clip_logits_map: Dict[str, List[np.ndarray]] = defaultdict(list)

    resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}

    for entry in tqdm(manifest_entries, desc="Exporting pseudo labels"):
        waveform, src_sr = torchaudio.load(entry.audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if src_sr != sample_rate:
            waveform = _get_resampler(resampler_cache, src_sr=src_sr, dst_sr=sample_rate)(waveform)

        starts = compute_segment_starts(
            num_samples=waveform.size(-1),
            window_samples=window_samples,
            hop_samples=hop_samples,
        )

        for segment_idx, start in enumerate(starts):
            stop = start + window_samples
            segment = waveform[:, start:stop]
            if segment.size(-1) < window_samples:
                segment = torch.nn.functional.pad(
                    segment,
                    (0, window_samples - segment.size(-1)),
                    mode="constant",
                    value=0.0,
                )

            model_input = _prepare_input(
                segment_waveform=segment,
                input_rank=input_rank,
                input_format=args.input_format,
                sample_rate=sample_rate,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                mel_hop_length=args.mel_hop_length,
                win_length=args.win_length,
                transpose_logmel=args.transpose_logmel,
            )

            output = session.run([output_name], {input_name: model_input})[0]
            logits = _extract_logits(output)
            segment_labels.append(
                SegmentPseudoLabel(
                    clip_id=entry.clip_id,
                    segment_idx=segment_idx,
                    logits=logits,
                )
            )
            clip_logits_map[entry.clip_id].append(logits)

    clip_labels: List[ClipPseudoLabel] = []
    for clip_id, logits_list in clip_logits_map.items():
        stacked = np.stack(logits_list, axis=0)
        agg_logits = stacked.mean(axis=0)
        probs = torch.softmax(torch.tensor(agg_logits, dtype=torch.float32), dim=0).numpy()
        hard = int(np.argmax(agg_logits))
        confidence = float(probs[hard])
        clip_labels.append(
            ClipPseudoLabel(
                clip_id=clip_id,
                agg_logits=agg_logits,
                hard_label=hard,
                confidence=confidence,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pseudo_labels_npz(output_path, segment_labels=segment_labels, clip_labels=clip_labels)

    num_classes = int(segment_labels[0].logits.shape[0]) if segment_labels else 0
    logging.info(
        "Saved pseudo labels to %s | clips=%d segments=%d classes=%d",
        output_path,
        len(clip_labels),
        len(segment_labels),
        num_classes,
    )


if __name__ == "__main__":
    main()
