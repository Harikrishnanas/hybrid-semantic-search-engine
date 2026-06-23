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
from collections import Counter
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

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "as", "from", "that", "this",
    "it", "its", "or", "and", "but", "not", "no", "so", "if", "then",
    "what", "which", "who", "when", "where", "how", "why", "there",
    "their", "they", "we", "our", "you", "your", "he", "she", "his",
    "her", "my", "i", "me", "us", "also", "more", "some", "any", "such",
    "these", "those", "each", "both", "all", "other", "than", "into",
    "through", "during", "before", "after", "above", "below",
}

_DOCUMENT_KEYWORDS = {
    "document", "text", "file", "pdf", "paper", "article", "chapter",
    "section", "content", "uploaded", "given", "provided", "mentioned",
    "stated", "according", "based", "context", "passage", "excerpt",
}



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
        "doc": "📄 Answered using uploaded document",
        "ai": "🧠 Answered using AI knowledge",
        "hybrid": "📄 + 🧠 Hybrid answer using document and AI knowledge",
        "fallback": "🧠 No supporting evidence found. Answered using AI knowledge.",
    }
    return mapping.get(source_type, "🧠 Answered using AI knowledge")


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


def _calculate_metrics(top_chunks: List[Dict[str, Any]]) -> Tuple[float, float, str, str]:
    """
    Returns (document_coverage, semantic_confidence, confidence_label, evidence_strength).
    """
    if not top_chunks:
        return 0.0, 0.0, "Very Low", "Very Weak"
        
    best_rerank = top_chunks[0].get("rerank_score", 0.0)
    
    if best_rerank <= 0.01:
        p = max(0, best_rerank) / 0.01
        semantic_confidence = p * 10
        confidence_label = "Very Low"
        evidence_strength = "Very Weak"
        document_coverage = 0.0
    elif best_rerank <= 0.05:
        p = (best_rerank - 0.01) / 0.04
        semantic_confidence = 10 + (p * 15)
        confidence_label = "Low"
        evidence_strength = "Weak"
        document_coverage = 25.0
    elif best_rerank <= 0.15:
        p = (best_rerank - 0.05) / 0.10
        semantic_confidence = 25 + (p * 25)
        confidence_label = "Moderate"
        evidence_strength = "Moderate"
        document_coverage = 50.0
    elif best_rerank <= 0.40:
        p = (best_rerank - 0.15) / 0.25
        semantic_confidence = 50 + (p * 25)
        confidence_label = "High"
        evidence_strength = "Strong"
        document_coverage = 75.0
    else:
        p = min(1.0, (best_rerank - 0.40) / 0.60)
        semantic_confidence = 75 + (p * 25)
        confidence_label = "Very High"
        evidence_strength = "Very Strong"
        document_coverage = 100.0
        
    return document_coverage, round(semantic_confidence, 1), confidence_label, evidence_strength


_GENERIC_TOPICS = {
    "example", "examples", "system", "systems", "data", "input", "output", 
    "information", "technology", "model", "models", "learning", "intelligence", 
    "analysis", "method", "methods", "approach", "results", "conclusion", 
    "introduction", "chapter", "section", "figure", "table", "process", 
    "overview", "summary", "background", "related work", "machine", "machines",
    "network", "networks", "algorithm", "algorithms"
}

_TOPIC_NORMS = {
    "Ai": "Artificial Intelligence",
    "Llm": "Large Language Models",
    "Llms": "Large Language Models",
    "Nlp": "Natural Language Processing",
    "Rag": "Retrieval-Augmented Generation",
    "Ml": "Machine Learning",
    "Dl": "Deep Learning",
    "Cv": "Computer Vision",
    "Generative Ai": "Generative AI",
    "Artificial Intelligence": "Artificial Intelligence",
    "Machine Learning": "Machine Learning",
    "Deep Learning": "Deep Learning"
}

def _get_topic_coverage(
    chunks: List[Dict[str, Any]], top_topics: List[str]
) -> List[CoverageItem]:
    """Calculate per-topic frequency as a percentage of total chunks."""
    if not top_topics or not chunks:
        return []

    topic_counts: Dict[str, int] = {t: 0 for t in top_topics}
    total = len(chunks)

    for chunk in chunks:
        text_lower = chunk["text"].lower()
        for topic in top_topics:
            topic_lower = topic.lower()
            if len(topic_lower) <= 3:
                # Use regex for short acronyms to avoid substring matching
                if re.search(r'\b' + re.escape(topic_lower) + r'\b', text_lower):
                    topic_counts[topic] += 1
            else:
                if topic_lower in text_lower:
                    topic_counts[topic] += 1

    items = [
        CoverageItem(topic=t, percentage=round(count / total * 100, 1))
        for t, count in topic_counts.items()
    ]
    items.sort(key=lambda x: x.percentage, reverse=True)
    return items


def _extract_top_topics(chunks: List[Dict[str, Any]], n: int = 10) -> List[str]:
    """Extract top N topics using headings and noun phrases, avoiding generic single words."""
    topics = []
    
    for chunk in chunks:
        text = chunk["text"]
        
        # 1. Extract Headings
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                heading = line.lstrip("# \t").strip()
                if 3 < len(heading) < 60:
                    topics.append(heading.title())
            elif 3 < len(line) < 60 and not line.endswith(('.', ',', '?', ':', ';')):
                words = line.split()
                if 1 <= len(words) <= 6:
                    if line.isupper() or all(w[0].isupper() for w in words if w.isalpha()):
                        topics.append(line.title())
        
        # 2. Extract multi-word title-case phrases (2 to 5 words)
        phrases = re.findall(r'\b(?:[A-Z][A-Za-z-]*\s+){1,4}[A-Z][A-Za-z-]*\b', text)
        for p in phrases:
            if len(p) < 60:
                topics.append(p.title())
                
        # 3. Extract common acronyms
        acronyms = re.findall(r'\b(?:AI|LLM|LLMs|NLP|RAG|ML|DL|CV)\b', text)
        for a in acronyms:
            topics.append(a.title())

    freq = Counter()
    for t in topics:
        t = " ".join(t.split())
        t_lower = t.lower()
        
        if t_lower in _GENERIC_TOPICS or t_lower in _STOPWORDS or t_lower in _DOCUMENT_KEYWORDS:
            continue
            
        # Ignore generic single words
        if len(t.split()) == 1 and t.title() not in _TOPIC_NORMS and t.upper() not in _TOPIC_NORMS:
            continue
            
        t_norm = _TOPIC_NORMS.get(t.title(), _TOPIC_NORMS.get(t.upper(), t.title()))
        freq[t_norm] += 1
        
    return [w for w, _ in freq.most_common(n)]


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
                evidence_strength="Very Weak",
                evidence_found=False,
                related_questions=[],
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

        # ── Coverage + Confidence ─────────────────────────────────────────
        document_coverage, semantic_confidence, confidence_label, evidence_strength = _calculate_metrics(final_results)

        # ── Generate Answer ───────────────────────────────────────────────
        if not final_results or document_coverage == 0.0:
            answer = LLMService.generate_general_answer(request.query)
            source_type = "ai"
            evidence_found = False
            document_coverage = 0.0
        else:
            answer = LLMService.generate_answer(request.query, final_results)
            if answer == "NOT_FOUND" or not answer.strip():
                answer = LLMService.generate_general_answer(request.query)
                source_type = "ai"
                evidence_found = False
                document_coverage = 0.0
            else:
                evidence_found = True
                source_type = "doc" if document_coverage >= 75.0 else "hybrid"

        # ── Related Questions (async-safe: just call LLM) ─────────────────
        context_snippet = " ".join(r["text"][:300] for r in final_results[:2]) if evidence_found else ""
        related_questions = LLMService.generate_related_questions(
            request.query, context_snippet
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
            related_questions=related_questions,
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
        top_topics = _extract_top_topics(chunks, n=10)
        heatmap = _get_topic_coverage(chunks, top_topics)

        return DocumentIntelligenceResponse(
            index_id=index_id,
            filename=filename or index_id,
            num_pages=0,   # unknown without re-reading PDF; caller provides via session
            num_chunks=len(chunks),
            top_topics=top_topics,
            coverage_heatmap=heatmap,
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
