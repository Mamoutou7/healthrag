# Created by Mamoutou Fofana
# Date: 10/24/2025

import spacy
from typing import List, Dict, Optional
from spacy.language import Language
from functools import lru_cache
from ..config import SCISPACY_MODEL

import warnings
warnings.filterwarnings('ignore')


class BioEntityExtractor:
    def __init__(self, model_name: str = SCISPACY_MODEL, disable_components: Optional[List[str]] = None):
        """
        Initialize the BioEntityExtractor with a sciSpaCy model.
        
        Args:
            model_name: Name of the sciSpaCy model to load
            disable_components: Optional list of pipeline components to disable for faster processing
        """
        try:
            # Disable unnecessary components for faster processing
            disable = disable_components or ["parser", "lemmatizer"]
            self.nlp: Language = spacy.load(model_name, disable=disable)
        except Exception as e:
            raise RuntimeError(f"Failed to load sciSpaCy model {model_name}: {str(e)}")

    @lru_cache(maxsize=1000)
    def _process_text(self, text: str) -> List[Dict]:
        """Process text and cache results to avoid redundant processing."""
        doc = self.nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]

    def extract(self, text: str) -> List[Dict]:
        """
        Extract entities from a single text string.
        
        Args:
            text: Input text to process
            
        Returns:
            List of dictionaries containing entity information
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if not text.strip():
            return []
        return self._process_text(text)

    def extract_entities_from_docs(self, docs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Extract entities from multiple documents using batch processing.
        
        Args:
            docs: List of dictionaries containing chunk_id and text
            
        Returns:
            Dictionary mapping chunk_id to list of extracted entities
        """
        if not docs:
            return {}
        
        # Validate input
        if not all(isinstance(doc, dict) and "chunk_id" in doc and "text" in doc for doc in docs):
            raise ValueError("Each document must be a dict with 'chunk_id' and 'text' keys")

        # Batch process texts for efficiency
        texts = [doc["text"] for doc in docs]
        chunk_ids = [doc["chunk_id"] for doc in docs]
        
        # Use spaCy's pipe for batch processing
        results = {}
        for chunk_id, doc_ents in zip(chunk_ids, self.nlp.pipe(texts)):
            results[chunk_id] = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc_ents.ents
            ]
        
        return results

if __name__ == "__main__":
    be = BioEntityExtractor()
    sample_text = "Remdesivir is used to treat severe COVID-19."
    print(be.extract(sample_text))