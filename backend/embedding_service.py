import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Singleton service to manage SentenceTransformer model loading and vector embedding generation.
    """
    _model_instance = None
    _model_name = "all-MiniLM-L6-v2"

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Retrieves the cached SentenceTransformer model instance or loads it if not already loaded.
        """
        if cls._model_instance is None:
            logger.info(f"Loading SentenceTransformer model: {cls._model_name}...")
            # Load model onto CPU/GPU as auto-detected by torch
            cls._model_instance = SentenceTransformer(cls._model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        return cls._model_instance

    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> np.ndarray:
        """
        Generates L2-normalized float32 embeddings for a list of input texts.
        Using normalized embeddings with inner product index (FAISS IndexFlatIP)
        provides direct cosine similarity scores.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            A numpy ndarray of shape (num_texts, embedding_dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = cls.get_model()
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # encode returns numpy arrays. We specify normalize_embeddings=True to compute L2 norm division
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Ensure correct type for FAISS (float32)
        embeddings_f32 = embeddings.astype(np.float32)
        logger.info(f"Embeddings generated with shape: {embeddings_f32.shape}")
        
        return embeddings_f32

    @classmethod
    def generate_query_embedding(cls, query: str) -> np.ndarray:
        """
        Generates a single L2-normalized float32 embedding for a query string.
        
        Args:
            query: Query string.
            
        Returns:
            A 1D numpy array of shape (embedding_dimension,) and dtype float32.
        """
        model = cls.get_model()
        
        # Encode a single text string
        embedding = model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embedding.astype(np.float32)
