"""Feature extraction helpers for student training."""

from __future__ import annotations

import torch
import torchaudio


def extract_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> torch.Tensor:
    """Convert waveform [1, T] into normalized log-mel [1, n_mels, frames]."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        center=center,
        power=2.0,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    mel = mel_transform(waveform)
    log_mel = amplitude_to_db(mel)

    mean = log_mel.mean(dim=(-1, -2), keepdim=True)
    std = log_mel.std(dim=(-1, -2), keepdim=True).clamp_min(1e-5)
    normalized = (log_mel - mean) / std
    return normalized
