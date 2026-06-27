"""
BM25 Retrieval Service
======================
Builds a BM25 index from chunk texts at indexing time, persists the tokenized
corpus alongside the existing FAISS metadata, and loads it for retrieval.

Depends only on:  rank-bm25  (lightweight, no heavy NLP)
"""

import os
import json
import logging
import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer (no NLTK needed)."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z][a-z0-9]{1,}\b", text)
    return tokens


class BM25Service:
    """
    Lightweight BM25 retrieval service.

    Files persisted per index_id:
        vector_store/{index_id}_bm25_corpus.json   — list of token lists
    """

    _vector_store_dir = "vector_store"
    _cache: Dict[str, BM25Okapi] = {}   # index_id → BM25Okapi object

    # ------------------------------------------------------------------ #

    @classmethod
    def _corpus_path(cls, index_id: str) -> str:
        os.makedirs(cls._vector_store_dir, exist_ok=True)
        return os.path.join(cls._vector_store_dir, f"{index_id}_bm25_corpus.json")

    # ------------------------------------------------------------------ #

    @classmethod
    def build_and_persist(cls, index_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Tokenize all chunk texts and persist the corpus for later BM25 search.

        Args:
            index_id: Unique document identifier (same as FAISS index_id).
            chunks:   List of chunk dicts with at least a "text" key.
        """
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]

        path = cls._corpus_path(index_id)
        logger.info(f"Persisting BM25 corpus ({len(tokenized_corpus)} docs) → {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenized_corpus, f)

        # Cache in memory
        cls._cache[index_id] = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built and cached for '{index_id}'.")

    # ------------------------------------------------------------------ #

    @classmethod
    def _load(cls, index_id: str) -> BM25Okapi:
        """Load BM25 index from cache or disk."""
        if index_id in cls._cache:
            return cls._cache[index_id]

        path = cls._corpus_path(index_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 corpus not found for index_id='{index_id}'. "
                f"Expected: {path}"
            )

        logger.info(f"Loading BM25 corpus from {path} …")
        with open(path, "r", encoding="utf-8") as f:
            tokenized_corpus = json.load(f)

        bm25 = BM25Okapi(tokenized_corpus)
        cls._cache[index_id] = bm25
        return bm25

    # ------------------------------------------------------------------ #

    @classmethod
    def search(
        cls,
        index_id: str,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Run BM25 retrieval and return top-k results with scores.

        Args:
            index_id: Document index identifier.
            query:    Raw query string.
            chunks:   Same chunk list used during indexing (for metadata lookup).
            top_k:    Number of results to return.

        Returns:
            List of dicts: {text, page_number, chunk_index, bm25_score, score}
        """
        bm25 = cls._load(index_id)
        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        # Pair each score with its chunk index and sort descending
        scored = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in scored:
            if score <= 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            results.append({
                "text": chunk["text"],
                "source_file": chunk.get("source_file", ""),
                "document_type": chunk.get("document_type", ""),
                "page_number": chunk.get("page_number"),
                "chunk_index": int(chunk["chunk_index"]),
                "bm25_score": float(score),
                "score": 0.0,
                "rerank_score": 0.0,
            })

        logger.info(
            f"BM25 search on '{index_id}': {len(results)} results (top_k={top_k})"
        )
        return results

    # ------------------------------------------------------------------ #

    @classmethod
    def corpus_exists(cls, index_id: str) -> bool:
        return os.path.exists(cls._corpus_path(index_id))
