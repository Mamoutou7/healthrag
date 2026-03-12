# -*- coding: utf-8 -*-
"""
Biomedical entity extraction with sciSpaCy.

- Extraction unitaire
- Extraction batch
- Sérialisation stable des entités
- Contrat de retour cohérent
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Dict, List, Optional

import spacy
from spacy.language import Language
from tqdm import tqdm

from ..config import SCISPACY_MODEL

warnings.filterwarnings("ignore")


class BioEntityExtractor:
    """
    Biomedical entity extractor using a sciSpaCy model.
    """

    def __init__(
        self,
        model_name: str = SCISPACY_MODEL,
        disable_components: Optional[List[str]] = None,
        batch_size: int = 32,
        n_process: int = 1,
        show_progress: bool = True,
    ) -> None:
        disable = disable_components or ["parser", "lemmatizer"]

        self.batch_size = batch_size
        self.n_process = n_process
        self.show_progress = show_progress

        try:
            self.nlp: Language = spacy.load(model_name, disable=disable)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sciSpaCy model '{model_name}': {exc}"
            ) from exc

    @staticmethod
    def _serialize_entities(doc) -> List[Dict]:
        """
        Convert spaCy entities into a stable list-of-dicts format.
        """
        return [
            {
                "text": ent.text,
                "normalized_text": ent.text.strip().lower(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    @staticmethod
    def chunk_entities_by_distance(
        entities: List[Dict],
        max_char_gap: int = 50,
        max_chunk_size: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        Group nearby entities into local clusters.

        Returns:
            {
                "chunk1": [...],
                "chunk2": [...]
            }
        """
        if not entities:
            return {}

        chunks: List[List[Dict]] = []
        current_chunk: List[Dict] = [entities[0]]

        for prev, curr in zip(entities, entities[1:]):
            char_gap = curr["start"] - prev["end"]

            if char_gap > max_char_gap or len(current_chunk) >= max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = [curr]
            else:
                current_chunk.append(curr)

        chunks.append(current_chunk)

        return {f"chunk{i + 1}": chunk for i, chunk in enumerate(chunks)}

    @lru_cache(maxsize=1000)
    def _process_text(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        return self._serialize_entities(doc)

    def extract(self, text: str) -> List[Dict]:
        """
        Extract flat entity list from a single text.

        Returns:
            [
                {
                    "text": ...,
                    "normalized_text": ...,
                    "label": ...,
                    "start": ...,
                    "end": ...
                }
            ]
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")

        text = text.strip()
        if not text:
            return []

        return self._process_text(text)

    def extract_chunked(
        self,
        text: str,
        max_char_gap: int = 50,
        max_chunk_size: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        Extract entities and group them by proximity for a single text.
        """
        entities = self.extract(text)
        return self.chunk_entities_by_distance(
            entities,
            max_char_gap=max_char_gap,
            max_chunk_size=max_chunk_size,
        )

    def extract_entities_from_docs(
        self,
        docs: List[Dict],
        max_char_gap: int = 50,
        max_chunk_size: int = 5,
        return_chunked: bool = False,
    ) -> Dict[str, List[Dict]] | Dict[str, Dict[str, List[Dict]]]:
        """
        Batch extract entities from docs.

        Args:
            docs: list of {"chunk_id": ..., "text": ...}
            return_chunked:
                - False -> {chunk_id: [entities]}
                - True  -> {chunk_id: {"chunk1": [...], ...}}

        Returns:
            Mapping indexed by chunk_id.
        """
        if not docs:
            return {}

        if not all(isinstance(doc, dict) and "chunk_id" in doc and "text" in doc for doc in docs):
            raise ValueError("Each document must be a dict with 'chunk_id' and 'text' keys")

        texts = [str(doc["text"]) for doc in docs]
        chunk_ids = [str(doc["chunk_id"]) for doc in docs]

        results = {}

        iterator = self.nlp.pipe(
            texts,
            batch_size=self.batch_size,
            n_process=self.n_process,
        )

        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=len(texts),
                desc="Extracting biomedical entities",
                ncols=80,
                unit="doc",
            )

        for chunk_id, doc_obj in zip(chunk_ids, iterator):
            entities = self._serialize_entities(doc_obj)

            if return_chunked:
                results[chunk_id] = self.chunk_entities_by_distance(
                    entities,
                    max_char_gap=max_char_gap,
                    max_chunk_size=max_chunk_size,
                )
            else:
                results[chunk_id] = entities

        return results


if __name__ == "__main__":
    extractor = BioEntityExtractor(batch_size=16, n_process=1)
    sample_text = (
        "Overexpression in metastatic breast cancer supports "
        "Syndecan-1 as a marker of invasiveness and poor prognosis."
    )
    print(extractor.extract(sample_text))
    print(extractor.extract_chunked(sample_text))