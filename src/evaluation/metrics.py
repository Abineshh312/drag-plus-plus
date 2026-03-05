"""DRAG++ evaluation metrics — accuracy, hallucination, latency, cost."""

from __future__ import annotations

from typing import List, Dict, Tuple
from dataclasses import dataclass
import time


@dataclass
class EvalMetrics:
    """Evaluation result metrics."""

    exact_match: float = 0.0
    f1_score: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    token_cost: float = 0.0
    num_samples: int = 0


class MetricsCalculator:
    """Compute DRAG++ evaluation metrics."""

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Exact match: 1 if prediction == reference (after normalize), 0 else.
        """
        pred_norm = " ".join(prediction.lower().split())
        ref_norm = " ".join(reference.lower().split())
        return 1.0 if pred_norm == ref_norm else 0.0

    @staticmethod
    def f1_score(prediction: str, reference: str) -> float:
        """
        Token-level F1 score.
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def aggregate(
        metrics_list: List[Tuple[float, float, float]],
    ) -> EvalMetrics:
        """
        Aggregate EM, F1, and hallucination rates across samples.

        Args:
            metrics_list: List of (em, f1, halluc_rate) tuples.

        Returns:
            Aggregated EvalMetrics.
        """
        if not metrics_list:
            return EvalMetrics()

        ems = [m[0] for m in metrics_list]
        f1s = [m[1] for m in metrics_list]
        halluc_rates = [m[2] for m in metrics_list]

        return EvalMetrics(
            exact_match=sum(ems) / len(ems),
            f1_score=sum(f1s) / len(f1s),
            hallucination_rate=sum(halluc_rates) / len(halluc_rates),
            num_samples=len(metrics_list),
        )
