# -*- coding: utf-8 -*-
"""
Created by: Mamoutou Fofana
Date: 2025-10-24
Description:
    BioEntityExtractor — sciSpaCy-based biomedical entity extractor
    with optimized caching and batch processing.
"""

import warnings
import spacy
from typing import List, Dict, Optional
from spacy.language import Language
from tqdm import tqdm
from functools import lru_cache
from ..config import SCISPACY_MODEL


warnings.filterwarnings('ignore')


class BioEntityExtractor:
    """
    Biomedical entity extractor using a sciSpaCy model.

    This class provides efficient single and batch entity extraction
    with caching, batch processing, and progress visualization.
    """

    def __init__(self,
                 model_name: str = SCISPACY_MODEL,
                 disable_components: Optional[List[str]] = None,
                 batch_size: int = 32,
                 n_process: int = 1,
                 show_progress: bool = True
                 ):
        """
        Initialize the BioEntityExtractor with a sciSpaCy model.
        
        Args:
            model_name: Name of the sciSpaCy model to load
            disable_components: Optional list of pipeline components to disable for faster processing
            batch_size (int): Batch size for nlp.pipe().
            n_process (int): Number of processes for parallelism (0 or 1 for CPU).
        """
        disable = disable_components or ["parser", "lemmatizer"]
        self.batch_size = batch_size
        self.n_process = n_process
        self.show_progress = show_progress

        try:
            self.nlp: Language = spacy.load(model_name, disable=disable)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sciSpaCy model {model_name}: {str(e)}"
            ) from e

    @staticmethod
    def _serialize_entities(doc) -> List[Dict[str, str]]:
        """Convert spaCy entities to a standardized dict format."""
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    @lru_cache(maxsize=1000)
    def _process_text(self,
                     text: str
                     ) -> List[Dict]:
        """Process and cache entity extraction for a single text."""
        doc = self.nlp(text)
        return self._serialize_entities(doc)

    def extract(self, text: str) -> List[Dict]:
        """
        Extract biomedical entities from a single text string.
        
        Args:
            text (str): Input text to process
            
        Returns:
            List[Dict[str, str]]: Extracted entities
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if not text.strip():
            return []
        return self._process_text(text)

    def extract_entities_from_docs(
            self,
            docs: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Extract entities from multiple documents using batch processing.

        Args:
            docs (List[Dict[str, str]]): List of dicts containing 'chunk_id' and 'text'.
            
        Returns:
            Dict[str, List[Dict[str, str]]]: Mapping of chunk_id to extracted entities.
        """
        if not docs:
            return {}
        
        # Validate input
        if not all(
                isinstance(doc, dict)
                and "chunk_id" in doc
                and "text" in doc
                for doc in docs
        ):
            raise ValueError(
                "Each document must be a dict with 'chunk_id' and 'text' keys"
            )

        # Batch process texts for efficiency
        texts = [doc["text"] for doc in docs]
        chunk_ids = [doc["chunk_id"] for doc in docs]
        
        # Use spaCy's pipe for batch processing
        results: Dict[str, List[Dict[str, str]]] = {}

        iterator = self.nlp.pipe(
            texts,
            batch_size=self.batch_size,
            n_process=self.n_process,
            disable=["parser", "lemmatizer"]
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
            results[chunk_id] = self._serialize_entities(doc_obj)

        
        return results

if __name__ == "__main__":
    be = BioEntityExtractor(batch_size=16, n_process=2)
    sample_text = "Overexpression in metastatic breast cancer supports Syndecan-1 as a marker of invasiveness and poor prognosis."
    print(be.extract(sample_text))