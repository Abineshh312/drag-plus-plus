"""Hybrid dense + sparse retrieval for DRAG++."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from loguru import logger


class HybridRetriever:
    """
    Hybrid retrieval combining dense (semantic) and sparse (BM25) search.
    
    Dense retrieval: semantic similarity via embeddings.
    Sparse retrieval: keyword matching via BM25.
    Combined: weighted average of both scores.
    """

    def __init__(
        self,
        documents: List[str],
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        """
        Initialize hybrid retriever.

        Args:
            documents: List of document texts to index.
            embedding_model: HuggingFace embedding model ID.
            dense_weight: Weight for dense (semantic) scores [0, 1].
            sparse_weight: Weight for sparse (BM25) scores [0, 1].
        """
        self.documents = documents
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        # Build dense embeddings
        self.embeddings = self.encoder.encode(documents, show_progress_bar=True)
        logger.info(f"Encoded {len(documents)} documents")

        # Build BM25 sparse index
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 index built")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k documents using hybrid scoring.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve.

        Returns:
            List of (doc_idx, doc_text, score) tuples, sorted by score descending.
        """
        # Dense retrieval
        query_embedding = self.encoder.encode(query)
        dense_scores = np.dot(self.embeddings, query_embedding)
        dense_scores = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-8
        )

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query), dtype=float)
        sparse_scores = (sparse_scores - sparse_scores.min()) / (
            sparse_scores.max() - sparse_scores.min() + 1e-8
        )

        # Hybrid score
        hybrid_scores = (
            self.dense_weight * dense_scores + self.sparse_weight * sparse_scores
        )

        # Get top-k
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        results = [
            (int(idx), self.documents[idx], float(hybrid_scores[idx]))
            for idx in top_indices
        ]

        return results
