"""
Multi-Document Hybrid Semantic Search Engine — FastAPI Backend
===============================================================
Pipeline:  BM25 + FAISS → Merge → CrossEncoder Rerank → Analytics
Supports:  PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, HTML, MD
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
from backend.document_processor import DocumentProcessor
from backend.embedding_service import EmbeddingService
from backend.faiss_service import FAISSService
from backend.bm25_service import BM25Service

# ─────────────────────────────────────────────────────────────────────────── #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reranker = CrossEncoder("BAAI/bge-reranker-base")

app = FastAPI(
    title="Multi-Document Hybrid Semantic Search Engine API",
    description=(
        "AI-Powered Knowledge Retrieval and Discovery Platform — "
        "BM25 + Dense Retrieval + CrossEncoder Reranking"
    ),
    version="3.0.0",
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

# Allowed extensions
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls",
    ".pptx", ".json", ".html", ".htm", ".md"
}
MAX_FILES = 20

# Strict relevance threshold on sigmoid-normalized rerank scores (0–1 scale)
# bge-reranker-base outputs raw logits; sigmoid converts to probability
EVIDENCE_THRESHOLD = 0.15

# ─────────────────────────────────────────────────────────────────────────── #
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────── #


def _classify_query(query: str, has_index: bool) -> str:
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


def _sigmoid(x: float) -> float:
    """Convert CrossEncoder raw logit to 0-1 probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _build_source_label(source_type: str) -> str:
    mapping = {
        "doc":    "📄 Document-Based Answer",
        "hybrid": "📄🧠 Document + AI Knowledge",
        "ai":     "🧠 AI Knowledge",
    }
    return mapping.get(source_type, "🧠 AI Knowledge")


def _merge_candidates(
    bm25_results: List[Dict[str, Any]],
    faiss_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge BM25 and FAISS candidates, deduplicate by chunk_index + source_file.
    """
    merged: Dict[Tuple, Dict[str, Any]] = {}

    for item in faiss_results:
        key = (item.get("source_file", ""), item["chunk_index"])
        merged[key] = {
            "text": item["text"],
            "source_file": item.get("source_file", ""),
            "document_type": item.get("document_type", ""),
            "page_number": item.get("page_number"),
            "chunk_index": item["chunk_index"],
            "score": item["score"],
            "bm25_score": 0.0,
            "rerank_score": 0.0,
        }

    for item in bm25_results:
        key = (item.get("source_file", ""), item["chunk_index"])
        if key in merged:
            merged[key]["bm25_score"] = item["bm25_score"]
        else:
            merged[key] = {
                "text": item["text"],
                "source_file": item.get("source_file", ""),
                "document_type": item.get("document_type", ""),
                "page_number": item.get("page_number"),
                "chunk_index": item["chunk_index"],
                "score": 0.0,
                "bm25_score": item["bm25_score"],
                "rerank_score": 0.0,
            }

    return list(merged.values())


def _calculate_semantic_confidence(
    best_rerank: float, best_faiss: float, best_bm25: float
) -> Tuple[float, str]:
    """
    Compute a weighted semantic confidence score from reranker, FAISS, and BM25.
    best_rerank should already be sigmoid-normalized (0-1).
    Returns (confidence_pct, label).
    """
    # Normalise BM25 to [0, 1] range (typical BM25 scores 0–20)
    norm_bm25 = min(best_bm25 / 20.0, 1.0)
    # best_rerank is already sigmoid-normalized (0-1)
    # best_faiss is cosine similarity (0-1)
    combined = (norm_bm25 * 0.20) + (best_faiss * 0.30) + (best_rerank * 0.50)
    combined = max(0.0, min(combined, 1.0))
    pct = round(combined * 100, 1)

    if pct >= 80:
        label = "Very High"
    elif pct >= 60:
        label = "High"
    elif pct >= 40:
        label = "Moderate"
    elif pct >= 20:
        label = "Low"
    else:
        label = "Very Low"

    return pct, label


def _calculate_document_coverage(
    answer: str,
    evidence_chunks: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    Sentence-level Grounded Document Coverage.
    Splits the generated answer into sentences and checks if each sentence 
    is supported by at least one evidence chunk using the CrossEncoder model.
    Coverage = (supported_sentences / total_sentences) * 100
    """
    import re
    if not answer or answer.strip() == "NOT_FOUND" or not evidence_chunks:
        return 0.0, "No Evidence"

    # 1. Split generated answer into sentences
    # Replace common bullets with periods for splitting, then split by punctuation
    clean_answer = re.sub(r"[\u2022\uf0b7\-]", ". ", answer)
    sentences = re.split(r"(?<=[.!?])\s+", clean_answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 0.0, "No Evidence"

    supported_sentences = 0
    total_sentences = len(sentences)

    # 2. Check support for each sentence
    for sentence in sentences:
        pairs = [(sentence, chunk["text"]) for chunk in evidence_chunks]
        if not pairs:
            continue
            
        # Use CrossEncoder to predict entailment/similarity between sentence and chunks
        raw_scores = reranker.predict(pairs)
        max_score = 0.0
        for raw in raw_scores:
            score = _sigmoid(float(raw))
            if score > max_score:
                max_score = score
                
        # If the best matching chunk for this sentence has a semantic score > 0.35, it's supported
        if max_score > 0.35:
            supported_sentences += 1

    # 3. Calculate coverage percentage
    coverage_pct = min(round((supported_sentences / total_sentences) * 100, 1), 100.0)

    # 4. Map to strict evidence strength labels
    if coverage_pct <= 20.0:
        strength = "No Evidence"
    elif coverage_pct <= 40.0:
        strength = "Weak Evidence"
    elif coverage_pct <= 60.0:
        strength = "Moderate Evidence"
    elif coverage_pct <= 80.0:
        strength = "Strong Evidence"
    else:
        strength = "Very Strong Evidence"

    return coverage_pct, strength


def _compute_source_attribution(
    final_results: List[Dict[str, Any]]
) -> Tuple[List[str], Dict[str, float]]:
    """
    Compute which source files contributed to the answer and their percentage
    of the supporting evidence.
    """
    if not final_results:
        return [], {}

    source_counts: Dict[str, int] = Counter(
        r["source_file"] for r in final_results if r.get("source_file")
    )
    total = sum(source_counts.values())
    if total == 0:
        return [], {}

    sources_used = list(source_counts.keys())
    evidence_distribution = {
        src: round((count / total) * 100, 1)
        for src, count in source_counts.items()
    }
    return sources_used, evidence_distribution


def _compute_document_intelligence(
    index_id: str, chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    from collections import defaultdict
    chunks_by_source = defaultdict(list)
    for chunk in chunks:
        src = chunk.get("source_file", "unknown")
        chunks_by_source[src].append(chunk)

    all_docs_intel = []
    
    for src, doc_chunks in chunks_by_source.items():
        safe_src = src.replace("/", "_").replace("\\", "_").replace(" ", "_")
        cache_path = os.path.join(UPLOAD_DIR, f"{index_id}_{safe_src}_intel.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    all_docs_intel.append(json.load(f))
                continue
            except Exception as e:
                logger.warning(f"Failed to load cached intelligence for {src}: {e}")

        # 1. LLM Concept Extraction for THIS document (30 candidate noun phrases)
        intel = LLMService.extract_document_intelligence(doc_chunks)
        summary = intel.get("summary", "")
        candidate_topics = intel.get("topics", [])
        
        doc_total_chunks = len(doc_chunks)
        
        # 2. Semantic Clustering of Candidate Topics
        import numpy as np
        topic_embeddings = []
        valid_topics = []
        for t in candidate_topics:
            if not t.strip(): continue
            try:
                emb = EmbeddingService.generate_query_embedding(t)
                topic_embeddings.append(emb)
                valid_topics.append(t)
            except Exception as e:
                logger.error(f"Error embedding topic '{t}': {e}")
                
        # Simple clustering: threshold = 0.80 cosine similarity
        clusters = [] 
        if valid_topics:
            embs_array = np.array(topic_embeddings)
            norms = np.linalg.norm(embs_array, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embs_norm = embs_array / norms
            sim_matrix = np.dot(embs_norm, embs_norm.T)
            
            visited = set()
            for i in range(len(valid_topics)):
                if i in visited: continue
                cluster = [valid_topics[i]]
                visited.add(i)
                for j in range(i+1, len(valid_topics)):
                    if j not in visited and sim_matrix[i, j] > 0.80:
                        cluster.append(valid_topics[j])
                        visited.add(j)
                clusters.append(cluster)
        
        # 3. Label each cluster (shortest phrase is usually the best semantic label)
        semantic_labels = []
        for cluster in clusters:
            label = sorted(cluster, key=len)[0]
            semantic_labels.append(label)
            
        # 4. Score true presence against chunks
        scored_topics = []
        for topic in semantic_labels:
            try:
                topic_emb = EmbeddingService.generate_query_embedding(topic)
                results = FAISSService.semantic_search(index_id, topic_emb, top_k=doc_total_chunks + 50)
                doc_results = [r for r in results if r.get("source_file") == src]
                
                topic_score = 0.0
                matched_chunks = 0
                for r in doc_results:
                    sim = r["score"]
                    if sim > 0.60:
                        topic_score += sim
                        matched_chunks += 1
                        
                if matched_chunks > 0:
                    presence = (matched_chunks / doc_total_chunks) * 100
                    scored_topics.append({
                        "topic": topic,
                        "topic_score": topic_score,
                        "percentage": round(presence, 1)
                    })
            except Exception as e:
                logger.error(f"Topic scoring error for '{topic}': {e}")

        scored_topics.sort(key=lambda x: x["topic_score"], reverse=True)
        final_top_10 = scored_topics[:10]
        
        logger.info(f"--- Document Intelligence Debug: {src} ---")
        for item in final_top_10:
            logger.info(f"Topic: {item['topic']} | Score: {item['topic_score']:.2f} | Presence: {item['percentage']}%")
            
        ordered_topics = [t["topic"] for t in final_top_10]
        heatmap_dicts = [{"topic": t["topic"], "percentage": t["percentage"]} for t in final_top_10]

        result = {
            "filename": src,
            "summary": summary,
            "top_topics": ordered_topics,
            "coverage_heatmap": heatmap_dicts,
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache intelligence: {e}")
            
        all_docs_intel.append(result)

    return all_docs_intel


def _load_index_meta(index_id: str) -> Dict[str, Any]:
    """Load the multi-doc index metadata (file list, types, pages, chunks)."""
    meta_path = os.path.join(UPLOAD_DIR, f"{index_id}_index_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_index_meta(index_id: str, meta: Dict[str, Any]) -> None:
    meta_path = os.path.join(UPLOAD_DIR, f"{index_id}_index_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────── #
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────── #


@app.post(
    "/upload-files",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents (up to 20).
    Supported: PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, HTML, MD
    """
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum {MAX_FILES} files allowed.",
        )

    saved_paths = []
    saved_names = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        safe_filename = os.path.basename(file.filename)
        dest_path = os.path.join(UPLOAD_DIR, safe_filename)

        try:
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(os.path.abspath(dest_path))
            saved_names.append(safe_filename)
            logger.info(f"Saved uploaded file: {safe_filename}")
        except Exception as e:
            logger.error(f"Failed to save file {safe_filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload {safe_filename}: {e}",
            )

    return UploadResponse(
        filenames=saved_names,
        file_paths=saved_paths,
        message=f"Successfully uploaded {len(saved_names)} file(s).",
    )


@app.post("/create-index", response_model=CreateIndexResponse)
async def create_index(request: CreateIndexRequest):
    """
    Process uploaded documents:
      1. Parse & chunk each document
      2. Embed all chunks → Single FAISS index
      3. Build BM25 corpus from all chunks
    """
    # Validate files exist
    for fp in request.file_paths:
        if not os.path.exists(fp):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {fp}",
            )

    try:
        # Use a combined index_id derived from sorted filenames
        filenames = sorted([os.path.basename(fp) for fp in request.file_paths])
        index_id = "multi_" + "_".join([os.path.splitext(f)[0][:8] for f in filenames])
        # Truncate to prevent too-long filenames
        index_id = index_id[:80]

        logger.info(f"Building combined index '{index_id}' for {len(request.file_paths)} files")

        all_chunks: List[Dict[str, Any]] = []
        file_stats = []
        total_pages = 0

        for file_path in request.file_paths:
            filename = os.path.basename(file_path)
            chunks = DocumentProcessor.process_document(file_path=file_path, filename=filename)

            if not chunks:
                logger.warning(f"No chunks extracted from {filename}, skipping.")
                continue

            # Track per-file page count
            page_nums = [c["page_number"] for c in chunks if c.get("page_number") is not None]
            num_pages = max(page_nums) if page_nums else 0
            total_pages += num_pages

            file_stats.append({
                "filename": filename,
                "document_type": chunks[0]["document_type"] if chunks else "Unknown",
                "num_chunks": len(chunks),
                "num_pages": num_pages,
            })
            logger.info(f"  ▸ {filename}: {len(chunks)} chunks extracted (type={chunks[0]['document_type']})")
            all_chunks.extend(chunks)

        logger.info(f"═══ INDEXING SUMMARY ═══")
        logger.info(f"  Total documents processed: {len(file_stats)}")
        for fs in file_stats:
            logger.info(f"  ▸ {fs['filename']}: {fs['num_chunks']} chunks, {fs['num_pages']} pages")
        logger.info(f"  Total chunks across all documents: {len(all_chunks)}")

        if not all_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the uploaded documents.",
            )

        # Embed all chunks → FAISS
        texts = [c["text"] for c in all_chunks]
        embeddings = EmbeddingService.generate_embeddings(texts)
        FAISSService.create_and_persist_index(
            index_id=index_id, embeddings=embeddings, chunks=all_chunks
        )

        # BM25 corpus
        BM25Service.build_and_persist(index_id=index_id, chunks=all_chunks)

        # Save index metadata
        meta = {
            "index_id": index_id,
            "filenames": [s["filename"] for s in file_stats],
            "document_types": list({s["document_type"] for s in file_stats}),
            "total_files": len(file_stats),
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "file_stats": file_stats,
        }
        _save_index_meta(index_id, meta)

        return CreateIndexResponse(
            index_id=index_id,
            total_files=len(file_stats),
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            message=f"Hybrid index (FAISS + BM25) created for {len(file_stats)} file(s).",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {e}",
        )


@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """
    Hybrid semantic search pipeline:
      BM25 + FAISS → Merge → CrossEncoder → Strict Threshold → Analytics → LLM Answer
    """
    t_start = time.time()

    try:
        # ── CASE 1: No document uploaded ─────────────────────────────────
        if not request.index_id:
            answer = LLMService.generate_knowledge_answer(request.query)
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
                sources_used=[],
                evidence_distribution={},
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

        # Load all chunks once (shared across concepts)
        _, all_chunks = FAISSService.load_index_and_metadata(request.index_id)
        total_chunks = len(all_chunks)

        # ── STEP 1: Query Decomposition ───────────────────────────────────
        concepts = LLMService.decompose_query(request.query)
        logger.info(f"═══ QUERY DECOMPOSITION: '{request.query[:80]}'")
        logger.info(f"  Concepts identified: {concepts}")

        # Detect if it's a definition-style query
        q_lower = request.query.lower().strip()
        def_prefixes = ["what is", "what are", "define", "meaning of", "explain"]
        is_def_query = any(q_lower.startswith(p) for p in def_prefixes)

        # ── STEP 2-4: Per-Concept Retrieval, Reranking & Context ─────────
        concept_contexts = {}   # concept → context string
        all_final_results = []  # flat pool for analytics & evidence display
        total_bm25 = 0
        total_faiss = 0
        total_merged = 0

        CHUNKS_PER_CONCEPT = max(3, min(request.top_k, 5))

        for concept in concepts:
            logger.info(f"  ── Retrieving for concept: '{concept}' ──")

            # BM25 for this concept
            bm25_res = BM25Service.search(
                index_id=request.index_id,
                query=concept,
                chunks=all_chunks,
                top_k=20,
            )
            total_bm25 += len(bm25_res)

            # FAISS for this concept
            concept_emb = EmbeddingService.generate_query_embedding(concept)
            faiss_res = FAISSService.semantic_search(
                index_id=request.index_id,
                query_embedding=concept_emb,
                top_k=20,
            )
            faiss_res = [r for r in faiss_res if r["score"] >= 0.10]
            total_faiss += len(faiss_res)

            # Merge
            merged_concept = _merge_candidates(bm25_res, faiss_res)
            total_merged += len(merged_concept)

            # CrossEncoder rerank for this concept
            if merged_concept:
                pairs = [(concept, m["text"]) for m in merged_concept]
                raw_scores = reranker.predict(pairs)
                for i, m in enumerate(merged_concept):
                    raw = float(raw_scores[i])
                    m["rerank_score"] = _sigmoid(raw)
                    m["raw_rerank_score"] = raw
                    m["concept"] = concept  # tag which concept this chunk belongs to

                # Boost definition chunks
                for m in merged_concept:
                    text_lower = m["text"].lower()
                    if ("is a " in text_lower or "is an " in text_lower
                            or "refers to" in text_lower
                            or "introduction" in text_lower
                            or "overview" in text_lower
                            or "definition" in text_lower):
                        m["rerank_score"] = min(m["rerank_score"] + 0.20, 1.0)

                merged_concept.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Keep top chunks for this concept
            top_for_concept = merged_concept[:CHUNKS_PER_CONCEPT]

            logger.info(f"    BM25={len(bm25_res)} FAISS={len(faiss_res)} Merged={len(merged_concept)}")
            for i, r in enumerate(top_for_concept):
                logger.info(
                    f"    #{i+1}: src={r.get('source_file','?')} chunk={r['chunk_index']} "
                    f"rerank={r['rerank_score']:.4f} faiss={r['score']:.4f}"
                )

            # Build concept-specific context block
            if top_for_concept:
                concept_text = "\n\n".join(r["text"] for r in top_for_concept)
                concept_contexts[concept] = concept_text
                all_final_results.extend(top_for_concept)

        # ── STEP 5: Build structured context (per-concept sections) ──────
        context_sections = []
        for concept, ctx_text in concept_contexts.items():
            context_sections.append(f"=== {concept} ===\n{ctx_text}")
        full_context = "\n\n".join(context_sections)

        logger.info(f"  Full context sections: {list(concept_contexts.keys())}")

        # ── Aggregate scores across all concepts ─────────────────────────
        best_rerank_score = max((r["rerank_score"] for r in all_final_results), default=0.0)
        best_faiss_score  = max((r["score"]        for r in all_final_results), default=0.0)
        best_bm25_score   = max((r["bm25_score"]   for r in all_final_results), default=0.0)

        # ── Semantic Confidence ───────────────────────────────────────────
        semantic_confidence, confidence_label = _calculate_semantic_confidence(
            best_rerank_score, best_faiss_score, best_bm25_score
        )

        # ── Filter for evidence threshold ─────────────────────────────────
        retrieved_chunks_count = len(all_final_results)
        evidence_results = [r for r in all_final_results if r["rerank_score"] >= EVIDENCE_THRESHOLD]

        has_real_evidence = best_rerank_score >= EVIDENCE_THRESHOLD

        if not has_real_evidence:
            semantic_confidence = min(semantic_confidence, 10.0)
            confidence_label = "Very Low"

        # ── Analytics ────────────────────────────────────────────────────
        elapsed_ms = round((time.time() - t_start) * 1000, 1)
        analytics = RetrievalAnalytics(
            total_chunks=total_chunks,
            bm25_matches=total_bm25,
            semantic_matches=total_faiss,
            merged_candidates=total_merged,
            final_results=len(evidence_results),
            response_time_ms=elapsed_ms,
        )

        # ── STEP 6: Determine Answer Mode & Generate Answer ───────────────
        # DOCUMENT  : best_rerank >= 0.55 — strong evidence
        # HYBRID    : best_rerank >= 0.35 — partial evidence
        # KNOWLEDGE : best_rerank <  0.35 — no useful evidence
        STRONG_THRESHOLD  = 0.55
        PARTIAL_THRESHOLD = 0.35

        if best_rerank_score >= STRONG_THRESHOLD and evidence_results:
            logger.info(f"  Answer Mode: DOCUMENT (rerank={best_rerank_score:.3f})")
            answer = LLMService.generate_document_answer(
                request.query, full_context, is_definition_query=is_def_query
            )
            if answer.strip() == "NOT_FOUND" or not answer.strip():
                logger.info("  DOCUMENT → NOT_FOUND, falling back to HYBRID")
                answer = LLMService.generate_hybrid_answer(
                    request.query, full_context, is_definition_query=is_def_query
                )
                source_type = "hybrid"
            else:
                source_type = "doc"
            evidence_found = True
            final_results_for_display = evidence_results

        elif best_rerank_score >= PARTIAL_THRESHOLD and evidence_results:
            logger.info(f"  Answer Mode: HYBRID (rerank={best_rerank_score:.3f})")
            answer = LLMService.generate_hybrid_answer(
                request.query, full_context, is_definition_query=is_def_query
            )
            source_type = "hybrid"
            evidence_found = True
            final_results_for_display = evidence_results

        else:
            logger.info(f"  Answer Mode: KNOWLEDGE (rerank={best_rerank_score:.3f})")
            answer = LLMService.generate_knowledge_answer(
                request.query, is_definition_query=is_def_query
            )
            source_type = "ai"
            evidence_found = False
            document_coverage = 0.0
            evidence_strength = "No Evidence"
            semantic_confidence = min(semantic_confidence, 10.0)
            confidence_label = "Very Low"
            final_results_for_display = []

        logger.info(f"  Answer Length: {len(answer)}")

        # ── Document Coverage (Sentence-level grounding) ──────────────────
        if has_real_evidence and evidence_results:
            document_coverage, evidence_strength = _calculate_document_coverage(
                answer=answer, evidence_chunks=evidence_results
            )
        else:
            document_coverage = 0.0
            evidence_strength = "No Evidence"

        # ── Source Attribution ────────────────────────────────────────────
        if evidence_found:
            sources_used, evidence_distribution = _compute_source_attribution(final_results_for_display)
        else:
            sources_used = []
            evidence_distribution = {}

        # ── Build result items ────────────────────────────────────────────
        result_items = [
            QueryResultItem(
                text=r["text"],
                score=r["score"],
                source_file=r.get("source_file", ""),
                document_type=r.get("document_type", ""),
                page_number=r.get("page_number"),
                chunk_index=r["chunk_index"],
                bm25_score=r["bm25_score"],
                rerank_score=r["rerank_score"],
            )
            for r in final_results_for_display
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
            sources_used=sources_used,
            evidence_distribution=evidence_distribution,
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
        logger.error(f"Query error, falling back to KNOWLEDGE mode: {e}", exc_info=True)
        try:
            answer = LLMService.generate_knowledge_answer(request.query)
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
                sources_used=[],
                evidence_distribution={},
                best_bm25_score=0.0,
                best_faiss_score=0.0,
                best_rerank_score=0.0,
                analytics=RetrievalAnalytics(
                    total_chunks=0, bm25_matches=0, semantic_matches=0,
                    merged_candidates=0, final_results=0, response_time_ms=elapsed
                ),
                best_match=None,
                results=[],
            )
        except Exception as fallback_e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search and fallback both failed: {fallback_e}",
            )


@app.get(
    "/document-intelligence/{index_id}",
    response_model=DocumentIntelligenceResponse,
)
async def document_intelligence(index_id: str):
    """
    Returns document intelligence: topics, topic presence, stats.
    """
    if not FAISSService.check_index_exists(index_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{index_id}' not found.",
        )

    try:
        _, chunks = FAISSService.load_index_and_metadata(index_id)
        intel = _compute_document_intelligence(index_id, chunks)
        meta = _load_index_meta(index_id)

        return DocumentIntelligenceResponse(
            index_id=index_id,
            total_files=meta.get("total_files", 1),
            total_pages=meta.get("total_pages", 0),
            total_chunks=len(chunks),
            document_types=meta.get("document_types", []),
            documents=intel,
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
        "service": "Multi-Document Hybrid Semantic Search Engine API",
        "version": "3.0.0",
    }
