from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def mae_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (yhat - y).abs().mean()

def mse_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(yhat, y)

def contrastive_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    '''
    InfoNCE-style contrastive loss aligning student features to teacher features.
    Implements the standard form referenced in Eq. (7) of the survey (anchors=student, positives=matching teacher, negatives=other samples).
    '''
    # Normalize
    s = F.normalize(student_feat, dim=-1)
    t = F.normalize(teacher_feat, dim=-1)

    logits = (s @ t.T) / temperature   # (B,B)
    labels = torch.arange(s.size(0), device=s.device)
    return F.cross_entropy(logits, labels)
