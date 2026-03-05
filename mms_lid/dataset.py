"""PyTorch dataset for MMS LID distillation with fixed-window segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from .features import extract_log_mel
from .manifest import ManifestEntry, load_manifest
from .pseudo_labels import PseudoLabelStore
from .segmentation import compute_segment_starts


@dataclass(frozen=True)
class SegmentRef:
    entry_idx: int
    segment_idx: int
    start_sample: int


class DistillSegmentDataset(Dataset):
    """Yields fixed-window segments with optional pseudo labels from teacher."""

    def __init__(
        self,
        manifest_path: str,
        split: Optional[str],
        sample_rate: int,
        window_sec: float,
        hop_sec: float,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_classes: int,
        pseudo_labels_path: Optional[str] = None,
        prefer_ground_truth: bool = True,
    ):
        self.entries: List[ManifestEntry] = load_manifest(manifest_path, split=split)
        self.sample_rate = int(sample_rate)
        self.window_samples = int(round(window_sec * sample_rate))
        self.hop_samples = int(round(hop_sec * sample_rate))
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.feature_hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.num_classes = int(num_classes)
        self.prefer_ground_truth = bool(prefer_ground_truth)

        if self.window_samples <= 0:
            raise ValueError("window_sec must produce positive window_samples")
        if self.hop_samples <= 0:
            raise ValueError("hop_sec must produce positive hop_samples")

        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}
        self.pseudo = PseudoLabelStore(pseudo_labels_path) if pseudo_labels_path else None

        self.segment_refs: List[SegmentRef] = self._build_segment_refs()

    def _get_resampler(self, from_sr: int) -> torchaudio.transforms.Resample:
        if from_sr not in self._resamplers:
            self._resamplers[from_sr] = torchaudio.transforms.Resample(
                orig_freq=from_sr,
                new_freq=self.sample_rate,
            )
        return self._resamplers[from_sr]

    def _estimate_num_samples_after_resample(self, num_frames: int, from_sr: int) -> int:
        if from_sr == self.sample_rate:
            return int(num_frames)
        return int(round(num_frames * float(self.sample_rate) / float(from_sr)))

    def _build_segment_refs(self) -> List[SegmentRef]:
        refs: List[SegmentRef] = []
        for entry_idx, entry in enumerate(self.entries):
            info = torchaudio.info(entry.audio_path)
            estimated_samples = self._estimate_num_samples_after_resample(
                num_frames=int(info.num_frames),
                from_sr=int(info.sample_rate),
            )
            starts = compute_segment_starts(
                num_samples=estimated_samples,
                window_samples=self.window_samples,
                hop_samples=self.hop_samples,
            )
            for segment_idx, start in enumerate(starts):
                refs.append(SegmentRef(entry_idx=entry_idx, segment_idx=segment_idx, start_sample=start))
        return refs

    def __len__(self) -> int:
        return len(self.segment_refs)

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        waveform, src_sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if src_sr != self.sample_rate:
            waveform = self._get_resampler(src_sr)(waveform)
        return waveform

    def __getitem__(self, index: int):
        ref = self.segment_refs[index]
        entry = self.entries[ref.entry_idx]

        waveform = self._load_waveform(entry.audio_path)

        start = ref.start_sample
        stop = start + self.window_samples
        segment = waveform[:, start:stop]

        if segment.size(-1) < self.window_samples:
            pad = self.window_samples - segment.size(-1)
            segment = torch.nn.functional.pad(segment, (0, pad), mode="constant", value=0.0)

        features = extract_log_mel(
            waveform=segment,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.feature_hop_length,
            win_length=self.win_length,
        )

        if self.prefer_ground_truth and entry.label is not None:
            hard_target = int(entry.label)
        elif self.pseudo is not None:
            hard_target = int(self.pseudo.get_clip_hard_label(entry.clip_id))
        else:
            raise ValueError(
                f"No hard target found for clip_id={entry.clip_id}. "
                "Provide labels in manifest or pseudo_labels_path."
            )

        if self.pseudo is not None and self.pseudo.has_segment(entry.clip_id, ref.segment_idx):
            teacher_logits = torch.tensor(
                self.pseudo.get_segment_logits(entry.clip_id, ref.segment_idx),
                dtype=torch.float32,
            )
            has_teacher = torch.tensor(True)
        else:
            teacher_logits = torch.zeros(self.num_classes, dtype=torch.float32)
            has_teacher = torch.tensor(False)

        return {
            "features": features,
            "hard_target": torch.tensor(hard_target, dtype=torch.long),
            "teacher_logits": teacher_logits,
            "has_teacher": has_teacher,
            "clip_id": entry.clip_id,
            "segment_idx": ref.segment_idx,
        }
