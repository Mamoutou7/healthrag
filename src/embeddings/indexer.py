# Created by Mamoutou Fofana
# Date: 10/25/2025


from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import (EMBED_MODEL, EMBEDDINGS_PICKLE, FAISS_INDEX_PATH,
                      MODELS_DIR)
from ..utils.input_output import load_pickle, save_pickle

MODELS_DIR.mkdir(parents=True, exist_ok=True)


class FaissIndexer:
    def __init__(self, model_name=EMBED_MODEL):
        """Initialize SentenceTransformer and FAISS index."""
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []

    def index_documents(self, documents: List[Dict]):
        texts = [doc["text"] for doc in documents]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Ensure correct dtype for FAISS
        if emb.dtype != np.float32:
            emb = emb.astype("float32")
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

    def save_index_metadata(
        self, index_path=FAISS_INDEX_PATH, metadata_path=EMBEDDINGS_PICKLE
    ):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(index_path))
        save_pickle(self.metadata, metadata_path)
        print(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")

    def load_index_metadata(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = EMBEDDINGS_PICKLE,
    ):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(str(index_path))
        self.metadata = load_pickle(metadata_path)
        print("FAISS index and metadata loaded successfully.")

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return top_k most similar text chunks for a given query."""
        if not len(self.metadata):
            raise ValueError("Index is empty. Please index or load data first.")

        query_vec = self.model.encode([query_text], convert_to_numpy=True).astype(
            "float32"
        )
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry["score"] = float(score)
                results.append(entry)
        return results
