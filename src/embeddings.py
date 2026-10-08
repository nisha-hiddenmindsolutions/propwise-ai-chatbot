"""
PropwiseAI - High-Performance Local Embedding Engine
Description:
    Provides local dense vector embeddings using SentenceTransformers (all-MiniLM-L6-v2).
    Eliminates external API key dependencies, rate limits, and token costs.
"""

import sys
import os
import logging
from typing import List

# Ensure parent project root is in Python path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

class LocalSentenceTransformerEmbeddings:
    """
    Direct SentenceTransformers embedding class compatible with LangChain FAISS VectorStore.
    Converts plain text strings into 384-dimensional dense float vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Initializing local SentenceTransformer embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model successfully initialized.")

    def embed_query(self, text: str) -> List[float]:
        """Converts a single query prompt into a vector representation."""
        if not text:
            text = "property"
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Converts a batch of document text chunks into vector representations."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def __call__(self, text: str) -> List[float]:
        """Allows direct callability as an embedding function."""
        return self.embed_query(text)

class EmbeddingManager:
    """
    Singleton class ensuring only ONE instance of the embedding model is loaded in memory.
    This saves CPU memory and avoids reloading model weights on every request.
    """
    _instance = None
    _embeddings_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._embeddings_model = LocalSentenceTransformerEmbeddings()
        return cls._instance

    @property
    def model(self) -> LocalSentenceTransformerEmbeddings:
        """Returns the active embedding model instance."""
        return self._embeddings_model

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings_model.embed_documents(texts)
