"""
DRAG++ Core Distillation Algorithm.

Novel contributions over DRAG (arxiv: 2506.01954):
1. Adaptive temperature scaling based on evidence confidence
2. Domain-shift detection for dynamic distillation weights
3. Contrastive evidence learning (positive/negative pairs)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Configuration for DRAG++ distillation."""
    base_temperature: float = 2.0
    min_temperature: float = 0.5
    max_temperature: float = 5.0
    alpha: float = 0.7          # Weight for distillation loss vs task loss
    beta: float = 0.2           # Weight for contrastive loss
    gamma: float = 0.1          # Weight for domain adaptation loss
    evidence_threshold: float = 0.6
    domain_shift_threshold: float = 0.3
    contrastive_margin: float = 0.5


class AdaptiveTemperatureScaler(nn.Module):
    """
    DRAG++ Innovation #1: Dynamically scales distillation temperature
    based on evidence confidence scores.

    Higher confidence evidence → lower temperature (sharper distributions)
    Lower confidence evidence → higher temperature (softer distributions)
    """

    def __init__(self, config: DistillationConfig) -> None:
        super().__init__()
        self.config = config
        self.confidence_net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, evidence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive temperature from evidence embeddings.

        Args:
            evidence_embeddings: [batch, hidden_dim] evidence representations.

        Returns:
            temperature: [batch, 1] per-sample temperatures.
        """
        confidence = self.confidence_net(evidence_embeddings)
        temperature = (
            self.config.max_temperature
            - confidence * (self.config.max_temperature - self.config.min_temperature)
        )
        return temperature.clamp(self.config.min_temperature, self.config.max_temperature)


class DomainShiftDetector(nn.Module):
    """
    DRAG++ Innovation #2: Detects domain shift and adjusts distillation
    weights dynamically to prevent knowledge transfer degradation.
    """

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute domain shift score between source and target.

        Returns:
            shift_score: [0, 1] domain distance.
            weight_adjustment: scalar to adjust distillation loss weight.
        """
        source_score = self.domain_classifier(source_embeddings).mean()
        target_score = self.domain_classifier(target_embeddings).mean()
        shift_score = torch.abs(source_score - target_score)
        weight_adjustment = float(1.0 - shift_score.item() * 0.5)
        return shift_score, weight_adjustment


class ContrastiveEvidenceLearner(nn.Module):
    """
    DRAG++ Innovation #3: Contrastive learning over positive/negative
    evidence pairs to sharpen retrieval-conditioned generation.
    """

    def __init__(self, hidden_dim: int = 768, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet contrastive loss for evidence discrimination.

        Args:
            anchor: Query embeddings [batch, hidden].
            positive: Relevant evidence embeddings [batch, hidden].
            negative: Irrelevant evidence embeddings [batch, hidden].

        Returns:
            Contrastive loss scalar.
        """
        a = F.normalize(self.projection(anchor), dim=-1)
        p = F.normalize(self.projection(positive), dim=-1)
        n = F.normalize(self.projection(negative), dim=-1)

        pos_dist = (a - p).pow(2).sum(dim=-1)
        neg_dist = (a - n).pow(2).sum(dim=-1)
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss


class DRAGPlusPlusDistiller(nn.Module):
    """
    DRAG++ complete distillation engine.

    Combines adaptive temperature, domain-shift detection,
    and contrastive evidence learning for superior RAG distillation.
    """

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        super().__init__()
        self.config = config or DistillationConfig()
        self.temp_scaler = AdaptiveTemperatureScaler(self.config)
        self.domain_detector = DomainShiftDetector()
        self.contrastive_learner = ContrastiveEvidenceLearner(
            margin=self.config.contrastive_margin
        )

    def kl_divergence_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive KL divergence with per-sample temperatures."""
        t = temperature.unsqueeze(-1)
        student_probs = F.log_softmax(student_logits / t, dim=-1)
        teacher_probs = F.softmax(teacher_logits / t, dim=-1)
        return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (t.mean() ** 2)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        evidence_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        pos_evidence: torch.Tensor,
        neg_evidence: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full DRAG++ distillation forward pass.

        Returns:
            Dictionary with total_loss and individual component losses.
        """
        # Adaptive temperature
        temperature = self.temp_scaler(evidence_embeddings)

        # KL distillation loss
        kl_loss = self.kl_divergence_loss(student_logits, teacher_logits, temperature)

        # Task loss (cross-entropy)
        task_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Domain shift adjustment
        _, weight_adj = self.domain_detector(source_embeddings, evidence_embeddings)

        # Contrastive evidence loss
        contrastive_loss = self.contrastive_learner(query_embeddings, pos_evidence, neg_evidence)

        # Weighted total loss
        alpha = self.config.alpha * weight_adj
        total_loss = (
            (1 - alpha) * task_loss
            + alpha * kl_loss
            + self.config.beta * contrastive_loss
        )

        return {
            "total_loss": total_loss,
            "kl_loss": kl_loss.detach(),
            "task_loss": task_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "mean_temperature": temperature.mean().detach(),
            "domain_weight_adj": torch.tensor(weight_adj),
        }
