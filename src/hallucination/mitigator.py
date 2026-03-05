"""DRAG++ Hallucination Mitigation — auto-correct via retrieval + regeneration."""

from __future__ import annotations

import torch
from typing import Optional
from loguru import logger


class HallucinationMitigator:
    """
    DRAG++ mitigation strategy: when hallucination is detected,
    re-retrieve evidence and regenerate with lower temperature.
    """

    def __init__(
        self,
        temperature_drop: float = 0.3,
        max_retries: int = 2,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.temperature_drop = temperature_drop
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

    def mitigate(
        self,
        query: str,
        original_answer: str,
        original_confidence: float,
        retriever,
        student_model,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Mitigate hallucination by re-retrieval + regeneration.

        Args:
            query: Original query.
            original_answer: Generated answer with hallucination.
            original_confidence: Detector confidence score.
            retriever: HybridRetriever instance.
            student_model: StudentModel instance.
            max_new_tokens: Max tokens for regeneration.

        Returns:
            Mitigated answer (re-retrieved + regenerated).
        """
        if original_confidence >= self.confidence_threshold:
            return original_answer  # No mitigation needed

        logger.warning(
            f"Hallucination detected (confidence={original_confidence:.3f}). "
            f"Attempting mitigation..."
        )

        for attempt in range(self.max_retries):
            # Re-retrieve with expansion
            expanded_query = f"{query}. What evidence supports this?"
            evidence = retriever.retrieve(expanded_query, top_k=10)

            if not evidence:
                logger.warning(f"Attempt {attempt+1}: No evidence found")
                continue

            # Build context from top evidence
            context = "\n".join([doc for _, doc, _ in evidence[:5]])

            # Regenerate with lower temperature
            lower_temp = max(0.1, 0.2 - self.temperature_drop * (attempt + 1))
            mitigated = student_model.generate(
                query,
                context,
                temperature=lower_temp,
                max_new_tokens=max_new_tokens,
            )

            logger.info(
                f"Attempt {attempt+1}: Regenerated with temp={lower_temp:.2f}"
            )
            return mitigated

        logger.error("Mitigation failed after all retries")
        return original_answer
