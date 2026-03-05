"""Loss definitions for offline KD training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    alpha: float,
    temperature: float,
    teacher_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KD objective with optional teacher masking."""
    ce = F.cross_entropy(student_logits, hard_targets)
    kl = torch.tensor(0.0, device=student_logits.device)

    has_teacher_signal = False
    if teacher_logits is not None:
        if teacher_mask is None:
            teacher_mask = torch.ones(
                student_logits.size(0), dtype=torch.bool, device=student_logits.device
            )

        if teacher_mask.any():
            has_teacher_signal = True
            selected_student = student_logits[teacher_mask]
            selected_teacher = teacher_logits[teacher_mask]
            kl = F.kl_div(
                F.log_softmax(selected_student / temperature, dim=1),
                F.softmax(selected_teacher / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)

    if has_teacher_signal:
        total = (1.0 - alpha) * ce + alpha * kl
    else:
        total = ce
    return total, ce.detach(), kl.detach()
