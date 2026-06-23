from backend.pdf_processor import PDFProcessor
from backend.embedding_service import EmbeddingService
import numpy as np

file_path = "uploads/resume HARIKRISHNAN A S.pdf"

# Show chunks
chunks = PDFProcessor.chunk_pdf(file_path)
print(f"Total chunks: {len(chunks)}\n")
for c in chunks:
    print(f"--- Chunk {c['chunk_index']} (page {c['page_number']}, {len(c['text'])} chars) ---")
    print(c["text"])
    print()

# Now test a search query
print("=" * 60)
print("SEARCH TEST: 'what are my projects done in sql'")
print("=" * 60)

texts = [c["text"] for c in chunks]
embeddings = EmbeddingService.generate_embeddings(texts)

query = "what are my projects done in sql"
query_emb = EmbeddingService.generate_query_embedding(query)

# Compute cosine similarity manually
scores = np.dot(embeddings, query_emb)
ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])

for score, chunk in ranked[:5]:
    print(f"\nScore: {score:.4f} | Page {chunk['page_number']} | Chunk {chunk['chunk_index']}")
    print(f"  {chunk['text'][:150]}...")
