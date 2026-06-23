from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ─────────────────────────────────────────────
#  Upload / Index
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    file_path: str = Field(..., description="Local path where the file is stored")
    message: str = Field(..., description="Status message")

class CreateIndexRequest(BaseModel):
    file_path: str = Field(..., description="Local path to the PDF file to index")

class CreateIndexResponse(BaseModel):
    index_id: str = Field(..., description="Unique identifier for the generated FAISS index")
    num_pages: int = Field(..., description="Total pages processed from the PDF")
    num_chunks: int = Field(..., description="Total text chunks generated and indexed")
    message: str = Field(..., description="Status message")

# ─────────────────────────────────────────────
#  Query Request
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Semantic search query string")
    top_k: int = Field(default=5, description="Number of final results to return")
    index_id: Optional[str] = Field(None, description="The ID of the FAISS index to query")

# ─────────────────────────────────────────────
#  Result Item — enriched with all scores
# ─────────────────────────────────────────────

class QueryResultItem(BaseModel):
    text: str = Field(..., description="The raw text content of the chunk")
    score: float = Field(..., description="FAISS cosine similarity score")
    page_number: int = Field(..., description="The source PDF page number (1-indexed)")
    chunk_index: int = Field(..., description="The index order of the chunk in the document")
    bm25_score: float = Field(default=0.0, description="BM25 retrieval score")
    rerank_score: float = Field(default=0.0, description="CrossEncoder reranker score")

# ─────────────────────────────────────────────
#  Retrieval Analytics
# ─────────────────────────────────────────────

class RetrievalAnalytics(BaseModel):
    total_chunks: int = Field(..., description="Total chunks in the index")
    bm25_matches: int = Field(..., description="Number of candidates from BM25")
    semantic_matches: int = Field(..., description="Number of candidates from FAISS")
    merged_candidates: int = Field(..., description="Unique candidates after merge")
    final_results: int = Field(..., description="Results after reranking")
    response_time_ms: float = Field(..., description="Total retrieval time in milliseconds")

# ─────────────────────────────────────────────
#  Document Intelligence
# ─────────────────────────────────────────────

class CoverageItem(BaseModel):
    topic: str
    percentage: float

class DocumentIntelligenceResponse(BaseModel):
    index_id: str
    filename: str
    num_pages: int
    num_chunks: int
    top_topics: List[str]
    coverage_heatmap: List[CoverageItem]

# ─────────────────────────────────────────────
#  Query Response — full enriched response
# ─────────────────────────────────────────────

class QueryResponse(BaseModel):
    answer: str
    source_type: str = Field(
        ...,
        description="'doc' | 'ai' | 'hybrid'"
    )
    query_type: str = Field(
        default="general",
        description="'document' | 'general' | 'mixed'"
    )
    source_label: str = Field(
        default="🧠 AI Knowledge",
        description="Human-readable source attribution label"
    )
    document_coverage: float = Field(
        default=0.0,
        description="0-100 document coverage score"
    )
    semantic_confidence: float = Field(
        default=0.0,
        description="0-100 semantic confidence score from reranker"
    )
    confidence_label: str = Field(
        default="Very Low",
        description="Label for semantic confidence"
    )
    evidence_strength: str = Field(
        default="No Evidence",
        description="No Evidence | Very Weak | Weak | Moderate | Strong | Very Strong"
    )
    evidence_found: bool = Field(
        default=False,
        description="Whether any supporting evidence was found in the document"
    )
    # Retrieval transparency scores
    best_bm25_score: float = Field(
        default=0.0,
        description="Best BM25 score from retrieval"
    )
    best_faiss_score: float = Field(
        default=0.0,
        description="Best dense similarity score from FAISS"
    )
    best_rerank_score: float = Field(
        default=0.0,
        description="Best CrossEncoder reranker score"
    )
    analytics: Optional[RetrievalAnalytics] = None
    best_match: Optional[QueryResultItem] = None
    results: List[QueryResultItem] = Field(default_factory=list)