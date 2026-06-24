"""
Hybrid Semantic Search Engine — FastAPI Backend
================================================
Pipeline:  BM25 + FAISS → Merge → CrossEncoder Rerank → Analytics
"""

import os
import re
import time
import shutil
import logging
import math
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

from backend.llm_service import LLMService
from backend.models import (
    UploadResponse,
    CreateIndexRequest,
    CreateIndexResponse,
    QueryRequest,
    QueryResultItem,
    QueryResponse,
    RetrievalAnalytics,
    DocumentIntelligenceResponse,
    CoverageItem,
)
from backend.pdf_processor import PDFProcessor
from backend.embedding_service import EmbeddingService
from backend.faiss_service import FAISSService
from backend.bm25_service import BM25Service

# ─────────────────────────────────────────────────────────────────────────── #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reranker = CrossEncoder("BAAI/bge-reranker-base")

app = FastAPI(
    title="Hybrid Semantic Search Engine API",
    description=(
        "AI-Powered Knowledge Retrieval and Discovery Platform — "
        "BM25 + Dense Retrieval + CrossEncoder Reranking"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────── #
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────── #





def _classify_query(query: str, has_index: bool) -> str:
    """
    Classify query as 'document', 'general', or 'mixed'.
    Heuristic: doc-referencing phrases → document; factual/world queries → general.
    """
    if not has_index:
        return "general"

    lower = query.lower()
    doc_indicators = [
        "in the document", "in this document", "according to", "in the paper",
        "in the text", "the document says", "based on", "what does the",
        "does the document", "the pdf", "in the file",
    ]
    general_indicators = [
        "what is", "how does", "explain", "define", "tell me about",
        "history of", "current", "latest", "recent", "world", "global",
        "country", "government", "economy", "science", "technology",
    ]
    doc_score = sum(1 for kw in doc_indicators if kw in lower)
    gen_score = sum(1 for kw in general_indicators if kw in lower)

    if doc_score > 0 and gen_score == 0:
        return "document"
    if gen_score > 0 and doc_score == 0:
        return "general"
    return "mixed"


def _build_source_label(source_type: str) -> str:
    mapping = {
        "doc":    "📄 Uploaded Document",
        "hybrid": "📄 + 🧠 Hybrid",
        "ai":     "🧠 AI Knowledge",
    }
    return mapping.get(source_type, "🧠 AI Knowledge")


def _merge_candidates(
    bm25_results: List[Dict[str, Any]],
    faiss_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge BM25 and FAISS candidates, deduplicate by chunk_index.
    If a chunk appears in both, take the max FAISS score and keep the BM25 score.
    """
    merged: Dict[int, Dict[str, Any]] = {}

    for item in faiss_results:
        ci = item["chunk_index"]
        merged[ci] = {
            "text": item["text"],
            "page_number": item["page_number"],
            "chunk_index": ci,
            "score": item["score"],
            "bm25_score": 0.0,
            "rerank_score": 0.0,
        }

    for item in bm25_results:
        ci = item["chunk_index"]
        if ci in merged:
            merged[ci]["bm25_score"] = item["bm25_score"]
        else:
            merged[ci] = {
                "text": item["text"],
                "page_number": item["page_number"],
                "chunk_index": ci,
                "score": 0.0,
                "bm25_score": item["bm25_score"],
                "rerank_score": 0.0,
            }

    return list(merged.values())


def _calculate_metrics(best_rerank: float) -> Tuple[float, float, str, str]:
    """
    Maps best_rerank score → (document_coverage, semantic_confidence, confidence_label, evidence_strength).
    All four values derive from the SAME rerank score so they are always consistent.
    """
    if best_rerank < 0.01:
        return 0.0, 5.0, "Very Low", "No Evidence"
    elif best_rerank < 0.05:
        return 0.0, 10.0, "Very Low", "Very Weak"
    elif best_rerank < 0.15:
        return 25.0, 30.0, "Low", "Weak"
    elif best_rerank < 0.35:
        return 50.0, 55.0, "Moderate", "Moderate"
    elif best_rerank < 0.60:
        return 75.0, 75.0, "High", "Strong"
    else:
        return 100.0, 95.0, "Very High", "Very Strong"


def _enforce_consistency(
    document_coverage: float,
    semantic_confidence: float,
    confidence_label: str,
    evidence_strength: str,
    source_type: str,
    evidence_found: bool,
) -> Tuple[float, float, str, str, str, bool]:
    """
    Guard: if coverage == 0 force all dependent metrics to their lowest values.
    This prevents impossible combinations like Coverage=0% + Confidence=Very High.
    """
    if document_coverage == 0.0 or not evidence_found:
        return 0.0, min(semantic_confidence, 10.0), "Very Low", "No Evidence", "ai", False
    return document_coverage, semantic_confidence, confidence_label, evidence_strength, source_type, evidence_found


def _compute_document_intelligence(index_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    cache_path = os.path.join(UPLOAD_DIR, f"{index_id}_intelligence.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached intelligence: {e}")
            
    # 1. LLM Concept Extraction
    intel = LLMService.extract_document_intelligence(chunks)
    summary = intel.get("summary", "")
    topics = intel.get("topics", [])
    
    if not topics:
        result = {"summary": summary, "top_topics": [], "coverage_heatmap": []}
        return result
        
    # 2. Semantic Density Coverage Algorithm
    topic_matches = {}
    total_matching_chunks = 0
    
    for topic in topics:
        try:
            topic_emb = EmbeddingService.generate_embedding(topic)
            # Fetch all chunks
            results = FAISSService.semantic_search(index_id, topic_emb, top_k=len(chunks))
            # Semantic Threshold: count chunks with cosine similarity > 0.35
            matches = sum(1 for r in results if r["score"] >= 0.35)
            topic_matches[topic] = matches
            total_matching_chunks += matches
        except Exception as e:
            logger.error(f"Coverage error for '{topic}': {e}")
            topic_matches[topic] = 0
            
    # 3. Calculate Relative Percentages
    heatmap_dicts = []
    if total_matching_chunks > 0:
        for topic, matches in topic_matches.items():
            if matches > 0:
                pct = round((matches / total_matching_chunks) * 100, 1)
                heatmap_dicts.append({"topic": topic, "percentage": pct})
    
    # Sort descending
    heatmap_dicts.sort(key=lambda x: x["percentage"], reverse=True)
    
    # Match topics list order to heatmap
    ordered_topics = [item["topic"] for item in heatmap_dicts]
    for topic in topics:
        if topic not in ordered_topics:
            ordered_topics.append(topic)
            
    result = {
        "summary": summary,
        "top_topics": ordered_topics,
        "coverage_heatmap": heatmap_dicts
    }
    
    # 4. Cache to disk
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to cache intelligence: {e}")
        
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────── #


@app.post(
    "/upload-pdf",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file. Validates extension and saves locally."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are allowed.",
        )

    safe_filename = os.path.basename(file.filename)
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        logger.info(f"Saving uploaded file {safe_filename} → {dest_path}")
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return UploadResponse(
            filename=safe_filename,
            file_path=os.path.abspath(dest_path),
            message="File uploaded successfully.",
        )
    except Exception as e:
        logger.error(f"Failed to save uploaded PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {e}",
        )


@app.post("/create-index", response_model=CreateIndexResponse)
async def create_index(request: CreateIndexRequest):
    """
    Process an uploaded PDF:
      1. Parse & chunk
      2. Embed chunks → FAISS index
      3. Build BM25 corpus and persist
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        )

    try:
        base_name = os.path.basename(request.file_path)
        index_id = os.path.splitext(base_name)[0]

        logger.info(f"Indexing: {request.file_path}")

        # 1. Chunk
        chunks = PDFProcessor.chunk_pdf(file_path=request.file_path)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from this document.",
            )

        # 2. Dense embeddings → FAISS
        texts = [c["text"] for c in chunks]
        embeddings = EmbeddingService.generate_embeddings(texts)
        FAISSService.create_and_persist_index(
            index_id=index_id, embeddings=embeddings, chunks=chunks
        )

        # 3. BM25 corpus
        BM25Service.build_and_persist(index_id=index_id, chunks=chunks)

        reader = PdfReader(request.file_path)
        num_pages = len(reader.pages)

        return CreateIndexResponse(
            index_id=index_id,
            num_pages=num_pages,
            num_chunks=len(chunks),
            message="Hybrid index (FAISS + BM25) created successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {e}",
        )


@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """
    Hybrid semantic search pipeline:
      BM25 + FAISS → Merge → CrossEncoder → Analytics → LLM Answer
    """
    t_start = time.time()

    try:
        # ── CASE 1: No document uploaded ─────────────────────────────────
        if not request.index_id:
            answer = LLMService.generate_general_answer(request.query)
            elapsed = (time.time() - t_start) * 1000
            return QueryResponse(
                answer=answer,
                source_type="ai",
                query_type="general",
                source_label=_build_source_label("ai"),
                document_coverage=0.0,
                semantic_confidence=0.0,
                confidence_label="Very Low",
                evidence_strength="No Evidence",
                evidence_found=False,
                best_bm25_score=0.0,
                best_faiss_score=0.0,
                best_rerank_score=0.0,
                analytics=RetrievalAnalytics(
                    total_chunks=0,
                    bm25_matches=0,
                    semantic_matches=0,
                    merged_candidates=0,
                    final_results=0,
                    response_time_ms=round(elapsed, 1),
                ),
                best_match=None,
                results=[],
            )

        # ── CASE 2: Document uploaded ─────────────────────────────────────
        if not FAISSService.check_index_exists(request.index_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index '{request.index_id}' not found. Process the document first.",
            )

        query_type = _classify_query(request.query, has_index=True)

        # Load all chunks (needed for BM25 + analytics)
        _, all_chunks = FAISSService.load_index_and_metadata(request.index_id)
        total_chunks = len(all_chunks)

        # ── BM25 retrieval ────────────────────────────────────────────────
        bm25_results = BM25Service.search(
            index_id=request.index_id,
            query=request.query,
            chunks=all_chunks,
            top_k=20,
        )

        # ── Dense (FAISS) retrieval ───────────────────────────────────────
        query_embedding = EmbeddingService.generate_query_embedding(request.query)
        faiss_results = FAISSService.semantic_search(
            index_id=request.index_id,
            query_embedding=query_embedding,
            top_k=20,
        )
        # Filter very weak FAISS matches
        faiss_results = [r for r in faiss_results if r["score"] >= 0.10]

        # ── Merge & deduplicate ───────────────────────────────────────────
        merged = _merge_candidates(bm25_results, faiss_results)

        # ── CrossEncoder Reranking ────────────────────────────────────────
        if merged:
            pairs = [(request.query, m["text"]) for m in merged]
            rerank_scores = reranker.predict(pairs)
            for i, m in enumerate(merged):
                m["rerank_score"] = float(rerank_scores[i])
            merged.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Keep top-k final results
        top_k = min(request.top_k, 5)
        final_results = merged[:top_k]

        # ── Retrieval transparency scores ─────────────────────────────────
        best_rerank_score = float(final_results[0]["rerank_score"]) if final_results else 0.0
        best_faiss_score  = max((r["score"]      for r in final_results), default=0.0)
        best_bm25_score   = max((r["bm25_score"] for r in final_results), default=0.0)

        # ── Analytics ────────────────────────────────────────────────────
        elapsed_ms = round((time.time() - t_start) * 1000, 1)
        analytics = RetrievalAnalytics(
            total_chunks=total_chunks,
            bm25_matches=len(bm25_results),
            semantic_matches=len(faiss_results),
            merged_candidates=len(merged),
            final_results=len(final_results),
            response_time_ms=elapsed_ms,
        )

        # ── Coverage + Confidence (single source of truth: best_rerank) ──
        document_coverage, semantic_confidence, confidence_label, evidence_strength = \
            _calculate_metrics(best_rerank_score)

        # ── Generate Answer ───────────────────────────────────────────────
        # Only attempt document-grounded answer if there is real evidence
        if document_coverage >= 25.0 and final_results:
            answer = LLMService.generate_answer(request.query, final_results)
            if answer == "NOT_FOUND" or not answer.strip():
                # LLM couldn't find it either — treat as AI
                answer = LLMService.generate_general_answer(request.query)
                source_type   = "ai"
                evidence_found = False
                document_coverage = 0.0
                semantic_confidence = min(semantic_confidence, 10.0)
                confidence_label = "Very Low"
                evidence_strength = "No Evidence"
            else:
                evidence_found = True
                source_type = "doc" if document_coverage >= 75.0 else "hybrid"
        else:
            answer = LLMService.generate_general_answer(request.query)
            source_type   = "ai"
            evidence_found = False

        # ── Consistency enforcement (final safety gate) ───────────────────
        document_coverage, semantic_confidence, confidence_label, evidence_strength, source_type, evidence_found = \
            _enforce_consistency(
                document_coverage, semantic_confidence, confidence_label,
                evidence_strength, source_type, evidence_found
            )

        # ── Build result items ────────────────────────────────────────────
        result_items = [
            QueryResultItem(
                text=r["text"],
                score=r["score"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                bm25_score=r["bm25_score"],
                rerank_score=r["rerank_score"],
            )
            for r in final_results
        ]

        return QueryResponse(
            answer=answer,
            source_type=source_type,
            query_type=query_type,
            source_label=_build_source_label(source_type),
            document_coverage=document_coverage,
            semantic_confidence=semantic_confidence,
            confidence_label=confidence_label,
            evidence_strength=evidence_strength,
            evidence_found=evidence_found,
            best_bm25_score=round(best_bm25_score, 4),
            best_faiss_score=round(best_faiss_score, 4),
            best_rerank_score=round(best_rerank_score, 4),
            analytics=analytics,
            best_match=result_items[0] if result_items else None,
            results=result_items,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search error: {e}",
        )


@app.get(
    "/document-intelligence/{index_id}",
    response_model=DocumentIntelligenceResponse,
)
async def document_intelligence(index_id: str, filename: str = ""):
    """
    Returns document intelligence: topics, coverage heatmap, stats.
    """
    if not FAISSService.check_index_exists(index_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{index_id}' not found.",
        )

    try:
        _, chunks = FAISSService.load_index_and_metadata(index_id)
        intel = _compute_document_intelligence(index_id, chunks)

        return DocumentIntelligenceResponse(
            index_id=index_id,
            filename=filename or index_id,
            num_pages=0,   # unknown without re-reading PDF; caller provides via session
            num_chunks=len(chunks),
            summary=intel.get("summary", ""),
            top_topics=intel.get("top_topics", []),
            coverage_heatmap=[CoverageItem(**item) for item in intel.get("coverage_heatmap", [])],
        )
    except Exception as e:
        logger.error(f"Document intelligence error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute document intelligence: {e}",
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Hybrid Semantic Search Engine API",
        "version": "2.0.0",
    }
