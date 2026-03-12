# -*- coding: utf-8 -*-
"""
Preprocess CSV PubMed data into chunked document JSON.

- Charge un CSV avec colonnes title / abstract / id
- Construit un champ texte
- Découpe le texte en chunks cohérents par phrase
- Sauvegarde le résultat en JSON
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import PROC_DIR, RAW_DIR
from ..utils.input_output import save_json

logger = logging.getLogger(__name__)


def load_pubmed_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a PubMed-like CSV and build a unified text field.

    Expected columns:
        - title
        - abstract
        - id (optional)

    Returns:
        pd.DataFrame with a 'text' column.
    """
    df = pd.read_csv(file_path)
    df = df.fillna("")

    if "title" not in df.columns:
        df["title"] = ""
    if "abstract" not in df.columns:
        df["abstract"] = ""

    df["text"] = (
        df["title"].astype(str).str.strip() + ". " + df["abstract"].astype(str).str.strip()
    ).str.strip()

    return df


def smart_chunk_text(
    text: str,
    max_words: int = 300,
    overlap_ratio: float = 0.15,
) -> List[str]:
    """
    Split text into sentence-aware chunks with overlap.

    Args:
        text: Input text.
        max_words: Max number of words per chunk.
        overlap_ratio: Fraction of previous chunk words reused in next chunk.

    Returns:
        List of chunk strings.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current_words: List[str] = []

    overlap_words = int(max_words * overlap_ratio)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = sentence.split()

        # Si une phrase seule dépasse max_words, on la découpe brutalement
        if len(sentence_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words).strip())
                current_words = []

            start = 0
            while start < len(sentence_words):
                end = start + max_words
                chunk_words = sentence_words[start:end]
                chunks.append(" ".join(chunk_words).strip())

                if overlap_words > 0:
                    start += max_words - overlap_words
                else:
                    start += max_words
            continue

        if current_words and len(current_words) + len(sentence_words) > max_words:
            chunks.append(" ".join(current_words).strip())

            if overlap_words > 0:
                current_words = current_words[-overlap_words:]
            else:
                current_words = []

        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words).strip())

    return chunks


def build_docs_from_pubmed(
    csv_path: Path,
    out_path: Path,
    max_words: int = 300,
    overlap_ratio: float = 0.15,
) -> List[Dict[str, str]]:
    """
    Build chunked documents from a PubMed-like CSV.

    Returns:
        List of dicts with:
            - doc_id
            - chunk_id
            - text
            - title
    """
    df = load_pubmed_csv(csv_path)
    docs: List[Dict[str, str]] = []

    for row_idx, row in df.iterrows():
        raw_doc_id = str(row.get("id", "")).strip()
        doc_id = raw_doc_id or f"pubmed_{row_idx}"

        chunks = smart_chunk_text(
            row["text"],
            max_words=max_words,
            overlap_ratio=overlap_ratio,
        )

        for chunk_idx, chunk_text in enumerate(chunks):
            docs.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{chunk_idx}",
                    "text": chunk_text,
                    "title": str(row.get("title", "")).strip(),
                }
            )

    save_json(docs, out_path)
    logger.info("Saved %s chunks to %s", len(docs), out_path)
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    build_docs_from_pubmed(RAW_DIR / "sample_pubmed.csv", PROC_DIR / "docs.json")