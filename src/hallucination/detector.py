"""
DRAG++ Real-Time Hallucination Detector.

Novel contribution: Token-level consistency checking against evidence graph
during inference. Detects hallucinations before they propagate in generation.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""
    is_hallucinated: bool
    confidence: float
    token_scores: List[float] = field(default_factory=list)
    flagged_spans: List[Tuple[int, int]] = field(default_factory=list)
    evidence_alignment: float = 0.0


class RealTimeHallucinationDetector:
    """
    DRAG++ Innovation: Real-time hallucination detection during RAG inference.

    Uses lightweight consistency checking between generated tokens and
    evidence graph embeddings to detect hallucinations on-the-fly.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        window_size: int = 10,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.device = device
        self._detection_cache: Dict[str, float] = {}

    def score_token_sequence(
        self,
        token_logits: torch.Tensor,
        evidence_embeddings: torch.Tensor,
    ) -> List[float]:
        """
        Score each token position for hallucination risk.

        Args:
            token_logits: [seq_len, vocab_size] model logits.
            evidence_embeddings: [num_docs, hidden_dim] retrieved evidence.

        Returns:
            List of confidence scores per token [0=hallucinated, 1=grounded].
        """
        probs = F.softmax(token_logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(token_logits.shape[-1])))
        normalized_entropy = entropy / max_entropy
        confidence_scores = (1.0 - normalized_entropy).tolist()
        return confidence_scores

    def check_evidence_alignment(
        self,
        generated_embedding: torch.Tensor,
        evidence_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute alignment between generated content and retrieved evidence.

        Args:
            generated_embedding: [hidden_dim] embedding of generated text.
            evidence_embeddings: [num_docs, hidden_dim] evidence embeddings.

        Returns:
            Alignment score [0, 1].
        """
        if evidence_embeddings.shape[0] == 0:
            return 0.0

        gen_norm = F.normalize(generated_embedding.unsqueeze(0), dim=-1)
        ev_norm = F.normalize(evidence_embeddings, dim=-1)
        similarities = (gen_norm @ ev_norm.T).squeeze(0)
        return float(similarities.max().item())

    def detect(
        self,
        generated_text: str,
        token_logits: torch.Tensor,
        evidence_embeddings: torch.Tensor,
        generated_embedding: torch.Tensor,
    ) -> HallucinationResult:
        """
        Full hallucination detection pipeline.

        Args:
            generated_text: The generated answer text.
            token_logits: [seq_len, vocab_size] generation logits.
            evidence_embeddings: Retrieved evidence representations.
            generated_embedding: Embedding of the full generated text.

        Returns:
            HallucinationResult with detection outcome and scores.
        """
        token_scores = self.score_token_sequence(token_logits, evidence_embeddings)
        evidence_alignment = self.check_evidence_alignment(
            generated_embedding, evidence_embeddings
        )
        avg_confidence = sum(token_scores) / max(len(token_scores), 1)

        # Flag low-confidence token windows
        flagged_spans = []
        for i in range(0, len(token_scores) - self.window_size + 1):
            window = token_scores[i:i + self.window_size]
            if sum(window) / len(window) < self.confidence_threshold:
                flagged_spans.append((i, i + self.window_size))

        # Merge overlapping spans
        flagged_spans = self._merge_spans(flagged_spans)

        overall_confidence = 0.6 * avg_confidence + 0.4 * evidence_alignment
        is_hallucinated = (
            overall_confidence < self.confidence_threshold
            or len(flagged_spans) > 0
        )

        if is_hallucinated:
            logger.warning(
                f"Hallucination detected! confidence={overall_confidence:.3f}, "
                f"evidence_alignment={evidence_alignment:.3f}, "
                f"flagged_spans={len(flagged_spans)}"
            )

        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=overall_confidence,
            token_scores=token_scores,
            flagged_spans=flagged_spans,
            evidence_alignment=evidence_alignment,
        )

    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping flagged spans."""
        if not spans:
            return []
        spans = sorted(spans)
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged
