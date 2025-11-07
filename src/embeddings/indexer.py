# -*- coding: utf-8 -*-
"""
FAISS + Sentence-Transformer indexer – robust, dimension-safe, cached.

Created by Mamoutou Fofana
Date: 2025-10-25

"""

from __future__ import annotations

import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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

warnings.filterwarnings('ignore')

# Logger configuration
logger = logging.getLogger(__name__)

# Ensure model cache directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class FaissIndexer:
    """FAISS wrapper that guarantees dimension consistency and lazy model loading."""
    def __init__(self, model_name: str = EMBED_MODEL):
        """Initialize SentenceTransformer and FAISS index."""
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.dim: Optional[int] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata:List[Dict[str, Any]] = []

    # Model handling (download once, cache forever)
    def _ensure_model(self) -> SentenceTransformer:
        """Download (if needed) and return the SentenceTransformer."""
        if self.model is not None:
            return self.model

        # Try a local cache folder first
        local_path = MODELS_DIR / self.model_name.split("/")[-1]
        try:
            if local_path.exists():
                logger.info(f"Loading from cache: {local_path}")
                self.model = SentenceTransformer(str(local_path), device="cpu", use_auth_token=True)

            else:
                logger.info(f"Downloading: {self.model_name}")
                download_dir = snapshot_download(
                    repo_id=self.model_name,
                    cache_dir=MODELS_DIR,
                    tqdm_class=None,
                )
                self.model = SentenceTransformer(download_dir, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Model download failed: {exc}"
            ) from exc

        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded – dimension: {self.dim}")
        return self.model


    def index_documents(self, documents: List[Dict]) -> None:
        """Encode texts and add them to the FAISS index."""
        _ = self._ensure_model()
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
            device="cpu"
        ).astype(np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array")

        current_dim = embeddings.shape[1]
        if self.dim is not None and current_dim != self.dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.dim}, got {current_dim}. "
            )

        # Create index on first batch
        if self.index is None:
            self.dim = current_dim
            self.index = faiss.IndexFlatL2(self.dim)
            logger.info(f"FAISS index created with dim={self.dim}")

        else:
            if self.index.d != self.dim:
                raise RuntimeError("FAISS index dimension corrupted.")

        self.index.add(embeddings)

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
        logger.info(f"Saved FAISS index to {index_path}")
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

        model = self._ensure_model()
        model_dim = model.get_sentence_embedding_dimension()

        if model_dim != index_dim:
            raise RuntimeError(
                f"Model dimension ({model_dim}) does not match FAISS index dimension ({index_dim})."
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

        _ = self._ensure_model()
        query_vec = self.model.encode(
            [query_text], 
            convert_to_numpy=True,
            device="cpu",
            batch_size=1
            ).astype(np.float32)

        if query_vec.ndim !=2 or query_vec.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dim ({query_vec.shape[1]}) != index dim ({self.dim}). "
            )

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        return results

if __name__=='__main__':
    documents = [
        {"doc_id": 1, "chunk_id": 0, "text": "Le chat dort sur le canapé."},
        {"doc_id": 1, "chunk_id": 1, "text": "Le chien joue dans le jardin."},
        {"doc_id": 2, "chunk_id": 0, "text": "Python est un langage de programmation populaire."},
        {"doc_id": 2, "chunk_id": 1, "text": "FAISS permet de rechercher efficacement dans des vecteurs."},
    ]

    indexer = FaissIndexer()
    print("Indexation...")
    indexer.index_documents(documents)
    print("Indexation terminée (chunks):", len(indexer.metadata))

    results = indexer.search("Comment rechercher rapidement dans des vecteurs ?", top_k=3)
    for r in results:
        print(f"{r['doc_id']} | {r['chunk_id']} | {r['score']:.2f} | {r['text']}")
