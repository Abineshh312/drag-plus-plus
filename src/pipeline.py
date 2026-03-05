"""DRAG++ end-to-end RAG inference pipeline."""

from __future__ import annotations

import time
from typing import Dict, Any
from loguru import logger

from .models.student import StudentModel
from .retrieval.retriever import HybridRetriever
from .hallucination.detector import RealTimeHallucinationDetector
from .hallucination.mitigator import HallucinationMitigator


class DRAGPlusPlusPipeline:
    """
    End-to-end DRAG++ inference pipeline:
    Query → Retrieve → Generate → Detect Hallucination → Mitigate (if needed).
    """

    def __init__(
        self,
        student_model: StudentModel,
        retriever: HybridRetriever,
        detector: RealTimeHallucinationDetector,
        mitigator: HallucinationMitigator,
    ) -> None:
        self.student = student_model
        self.retriever = retriever
        self.detector = detector
        self.mitigator = mitigator

    def run(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Full DRAG++ pipeline: retrieve → generate → detect → optionally mitigate.

        Returns:
            Dictionary with answer, latency, hallucination score, etc.
        """
        start_time = time.time()

        # 1. Retrieve
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        context = "\n".join([doc for _, doc, _ in retrieved])

        logger.info(f"Retrieved {len(retrieved)} documents")

        # 2. Generate
        answer = self.student.generate(
            query, context, temperature=temperature, max_new_tokens=max_new_tokens
        )

        # 3. Detect hallucination
        # (Simplified: use confidence based on answer length + retrieval scores)
        avg_retrieval_score = sum(score for _, _, score in retrieved) / len(retrieved)
        halluc_confidence = min(0.9, 0.5 + 0.4 * avg_retrieval_score)

        result = {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieved": retrieved,
            "hallucination_confidence": halluc_confidence,
            "was_mitigated": False,
            "latency_ms": (time.time() - start_time) * 1000,
        }

        # 4. Mitigate if needed
        if halluc_confidence < self.detector.confidence_threshold:
            logger.warning(f"Low confidence ({halluc_confidence:.3f}). Mitigating...")
            mitigated = self.mitigator.mitigate(
                query, answer, halluc_confidence, self.retriever, self.student
            )
            result["answer"] = mitigated
            result["was_mitigated"] = True
            result["latency_ms"] = (time.time() - start_time) * 1000

        logger.info(f"Pipeline complete: {result['latency_ms']:.1f}ms")
        return result
