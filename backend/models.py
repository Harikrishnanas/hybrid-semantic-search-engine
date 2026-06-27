from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ─────────────────────────────────────────────
#  Upload / Index
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    filenames: List[str] = Field(..., description="The names of the uploaded files")
    file_paths: List[str] = Field(..., description="Local paths where the files are stored")
    message: str = Field(..., description="Status message")

class CreateIndexRequest(BaseModel):
    file_paths: List[str] = Field(..., description="Local paths to the files to index")

class CreateIndexResponse(BaseModel):
    index_id: str = Field(..., description="Unique identifier for the generated FAISS index")
    total_files: int = Field(..., description="Total files processed")
    total_pages: int = Field(..., description="Total pages/sections processed")
    total_chunks: int = Field(..., description="Total text chunks generated and indexed")
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
    source_file: str = Field(..., description="The original filename")
    document_type: str = Field(..., description="The type of the document")
    page_number: Optional[int] = Field(None, description="The source PDF page number (1-indexed)")
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

class DocumentIntelligenceItem(BaseModel):
    filename: str
    summary: str
    top_topics: List[str]
    coverage_heatmap: List[CoverageItem]

class DocumentIntelligenceResponse(BaseModel):
    index_id: str
    total_files: int
    total_pages: int
    total_chunks: int
    document_types: List[str] = Field(default_factory=list)
    documents: List[DocumentIntelligenceItem] = Field(default_factory=list)

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
    sources_used: List[str] = Field(
        default_factory=list,
        description="List of filenames used to support the answer"
    )
    evidence_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of evidence contributed by each source file"
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