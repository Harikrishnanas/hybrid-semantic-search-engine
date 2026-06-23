import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSService:
    """
    Handles FAISS index creation, persistence, on-demand loading with caching, and semantic search.
    """
    _vector_store_dir = "vector_store"
    # Simple cache to hold loaded indexes and metadata in-memory: {index_id: (faiss_index, metadata_list)}
    _index_cache: Dict[str, Tuple[faiss.Index, List[Dict[str, Any]]]] = {}

    @classmethod
    def get_paths(cls, index_id: str) -> Tuple[str, str]:
        """
        Helper to return the paths for the FAISS index file and the metadata JSON file.
        """
        os.makedirs(cls._vector_store_dir, exist_ok=True)
        index_path = os.path.join(cls._vector_store_dir, f"{index_id}.index")
        metadata_path = os.path.join(cls._vector_store_dir, f"{index_id}_metadata.json")
        return index_path, metadata_path

    @classmethod
    def create_and_persist_index(
        cls, index_id: str, embeddings: np.ndarray, chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Creates a FAISS IndexFlatIP (Inner Product), adds the normalized embeddings,
        saves the index and chunk metadata locally, and caches it in memory.
        
        Args:
            index_id: A unique identifier for the PDF document index.
            embeddings: Float32 NumPy array of normalized embeddings.
            chunks: List of chunk metadata dictionaries (text, page_number, chunk_index).
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch between number of embeddings ({len(embeddings)}) and chunks ({len(chunks)})")

        if len(embeddings) == 0:
            logger.warning("Empty embeddings array. Skipping index creation.")
            return

        dimension = embeddings.shape[1]
        index_path, metadata_path = cls.get_paths(index_id)

        logger.info(f"Creating FAISS index flat-IP for {index_id} with dimension {dimension}")
        # IndexFlatIP with L2 normalized vectors calculates Cosine Similarity directly
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Write index to disk
        logger.info(f"Saving FAISS index file to {index_path}...")
        faiss.write_index(index, index_path)

        # Write chunk metadata to disk
        logger.info(f"Saving metadata JSON file to {metadata_path}...")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # Cache in memory
        cls._index_cache[index_id] = (index, chunks)
        logger.info(f"Successfully created and persisted index '{index_id}' with {len(chunks)} chunks.")

    @classmethod
    def load_index_and_metadata(
        cls, index_id: str
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Loads the FAISS index and chunk metadata from disk or cache.
        
        Args:
            index_id: The identifier for the index to load.
            
        Returns:
            A tuple of (faiss_index_object, list_of_metadata_dicts)
        """
        # Return from cache if present
        if index_id in cls._index_cache:
            logger.info(f"Retrieving index '{index_id}' from in-memory cache.")
            return cls._index_cache[index_id]

        index_path, metadata_path = cls.get_paths(index_id)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Index files not found for index_id '{index_id}'. "
                f"Expected index: {index_path}, metadata: {metadata_path}"
            )

        # Read FAISS index
        logger.info(f"Reading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)

        # Read metadata JSON
        logger.info(f"Reading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Cache it
        cls._index_cache[index_id] = (index, chunks)
        return index, chunks

    @classmethod
    def semantic_search(
        cls, index_id: str, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Performs vector search in the FAISS index and retrieves top matching chunks with metadata.
        
        Args:
            index_id: Index identifier.
            query_embedding: L2 normalized 1D float32 numpy array of the query embedding.
            top_k: Number of results to retrieve.
            
        Returns:
            A list of retrieved search result dicts containing matching text, score, page number, and chunk index:
            [
                {
                    "text": "...",
                    "score": 0.85,
                    "page_number": 2,
                    "chunk_index": 12
                },
                ...
            ]
        """
        index, chunks = cls.load_index_and_metadata(index_id)

        # Reshape query embedding for FAISS index search (needs to be 2D array of shape [1, dimension])
        query_vector = np.expand_dims(query_embedding, axis=0)

        # Perform search. distances contains inner products (which are cosine similarity scores)
        # and indices contains indices of matching vectors
        distances, indices = index.search(query_vector, top_k)

        results = []
        # distances[0] and indices[0] correspond to query index 0 (our only query)
        for score, idx in zip(distances[0], indices[0]):
            # -1 is returned by FAISS if there are not enough items in index to return top_k
            if idx == -1:
                continue
            
            # Bound check safety
            if idx < 0 or idx >= len(chunks):
                logger.error(f"FAISS index {idx} out of range of chunks list (length {len(chunks)})")
                continue

            metadata = chunks[idx]
            
            # For cosine similarity, FAISS flat IP returns dot product. Since both vectors are unit length (L2 norm = 1),
            # this represents exact cosine similarity.
            results.append({
                "text": metadata["text"],
                "score": float(score),
                "page_number": int(metadata["page_number"]),
                "chunk_index": int(metadata["chunk_index"])
            })

        logger.info(f"Semantic search completed. Found {len(results)} matches for index '{index_id}'.")
        return results

    @classmethod
    def check_index_exists(cls, index_id: str) -> bool:
        """
        Checks if the FAISS index and metadata files exist on disk for the given index_id.
        """
        index_path, metadata_path = cls.get_paths(index_id)
        return os.path.exists(index_path) and os.path.exists(metadata_path)
