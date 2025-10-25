# Created by Mamoutou Fofana
# Date: 10/25/2025

"""
FAISS + Sentence-Transformer indexer – robust, dimension-safe, cached.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from ..config import (
    EMBED_MODEL,
    EMBEDDINGS_PICKLE,
    FAISS_INDEX_PATH,
    MODELS_DIR
)
from ..utils.input_output import load_pickle, save_pickle


# Logger 
logger = logging.getLogger(__name__)

# Ensure model cache directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class FaissIndexer:
    """FAISS wrapper that guarantees dimension consistency and lazy model loading."""
    def __init__(self, model_name: str = EMBED_MODEL):
        """Initialize SentenceTransformer and FAISS index."""
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.dim: int | None = None
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata:List[Dict[str, Any]] = []


    # Model handling (download once, cache forever)
    def _ensure_model(self) -> SentenceTransformer:
        """Download (if needed) and return the SentenceTransformer."""
        if self.model is not None:
            return self.model

        # Try a local cache folder first
        local_path = MODELS_DIR / self.model_name.split("/")[-1]
        if local_path.exists():
            log.info(f"Loading SentenceTransformer from local cache: {local_path}")
            self.model = SentenceTransformer(str(local_path), device="cpu")

        else:
            logger.info(f"Downloading SentenceTransformer model: {self.model_name}")
            try:
                download_dir = snapshot_download(
                    repo_id=self.model_name,
                    cache_dir=MODELS_DIR,
                    tqdm_class=None,          # silence progress bar in prod
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not download model '{self.model_name}'. "
                    "Check your HF_TOKEN or switch to a public model. "
                    f"({e})"
                ) from e
            self.model = SentenceTransformer(download_dir, device="cpu")

        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded – dimension: {self.dim}")

        return self.model


    # Index building
    def index_documents(self, documents: List[Dict]) -> None:
        """Encode texts and add them to the FAISS index."""
        model = self._ensure_model()
        texts = [doc["text"] for doc in documents]
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype(np.float32)

        current_dim = emb.shape[1]
        if self.dim is not None and current_dim != self.dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim}, got {current_dim}. "
                "All documents must be encoded with the same model."
        )

        # Create index on first batch
        if self.index is None:
            self.dim = current_dim
            self.index = faiss.IndexFlatL2(self.dim)
            logger.info(f"FAISS index created with dim={self.dim}")

        else:
            if self.index.d != self.dim:
                raise RuntimeError("FAISS index dimension corrupted.")

        self.index.add(emb)

        self.metadata.extend(
            [
                {
                    "doc_id": doc.get("doc_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "text": doc.get("text"),
                }
                for doc in documents
            ]
        )
        logger.info(f"Indexed {len(documents)} chunks (total={len(self.metadata)})")

    def save_index_metadata(
        self,
        index_path=FAISS_INDEX_PATH,
        metadata_path=EMBEDDINGS_PICKLE
    ) -> None:
        """Save FAISS index and metadata."""
        if self.index is None or not self.metadata:
            raise ValueError("Nothing to save - index is empty.")
        faiss.write_index(self.index, str(index_path))
        save_pickle(self.metadata, metadata_path)
        looger.info(f"Saved FAISS index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")


    def load_index_metadata(
        self,
        index_path = FAISS_INDEX_PATH,
        metadata_path = EMBEDDINGS_PICKLE
    ) -> None:
        """Load FAISS index and metadata (used by FastAPI startup)."""
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata pickle not found: {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = load_pickle(metadata_path)
        index_dim = self.index.d 

        # Charger le modèle pour vérifier la dimension
        model = self._ensure_model()
        model_dim = model.get_sentence_embedding_dimension()

        if model_dim != index_dim:
            raise RuntimeError(
                f"Model dimension ({model_dim}) does not match FAISS index dimension ({index_dim}).\n"
                "→ You likely indexed with a different model than the one currently loaded.\n"
                "→ Fix: Re-index all documents with the current model, or use the correct model."
            )

        self.dim = index_dim
        logger.info(f"Loaded FAISS index ({len(self.metadata)} chunks) from {index_path}")


    def search(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top_k most similar text chunks for a given query."""
        if not len(self.metadata):
            raise ValueError("Index is empty. Please index or load data first.")

        if self.index is None or self.dim is None:
            raise RuntimeError("FAISS index not initialized.")

        model = self._ensure_model()
        query_vec = self.model.encode(
            [query_text], 
            convert_to_numpy=True
            ).astype(np.float32)

        # Vérification critique
        if query_vec.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dim ({query_vec.shape[1]}) != index dim ({self.dim}). "
                "Use the same model for indexing and querying."
            )

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        return results
