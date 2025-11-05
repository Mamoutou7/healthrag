# -*- coding: utf-8 -*-
"""
Created by: Mamoutou Fofana
Date: 2025-10-24
Description:
    Script to preprocess CSV data into chunked document JSONs.
"""

from pathlib import Path
import pandas as pd
import re
import logging
from typing import List, Dict
from ..utils.input_output import save_json
from ..config import RAW_DIR, PROC_DIR



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def load_pubmed_csv(file_path: Path):
	"""
	Load a CSV file and prepare text data
	Args:
		file_path: Path to the CSV file
	Returns:
		pd.DataFrame: DataFrame with 'text' column combining title
	"""
	df = pd.read_csv(file_path)
	df = df.fillna("")
	df["text"] = df["title"].astype(str) + "." + df["abstract"].astype(str)
	return df 


def smart_chunk_text(text: str,
					 max_words: int = 300,
					 overlap_ratio: float = 0.15
					  ) -> List[str]:
	"""
	Optimal text chunking by sentences with smart overlap.
	Args:
		text (str): Input text to split.
		max_words (int): Maximum number of words per chunk.
		overlap_ratio (float): Fraction of overlap between chunks.
	Returns:
        List[str]: List of coherent text chunks.
	"""
	text = re.sub(r"\s", " ", text).strip()
	sentences = re.split(r'(?<=[.!?])\s+', text)
	chunks = []
	current_chunk = []

	overlap_words = int(max_words * overlap_ratio)
	word_count = 0

	for sentence in sentences:
		sentence_words = sentence.split()
		sentence_len = len(sentence_words)

		if word_count + sentence_len > max_words:
			chunks.append(" ".join(current_chunk).strip())

			if overlap_words > 0 and current_chunk:
				overlap = " ".join(current_chunk[-overlap_words:])
				current_chunk = overlap.split()
				word_count = len(current_chunk)
			else:
				current_chunk = []
				word_count = 0

		current_chunk.extend(sentence_words)
		word_count += sentence_len

		if current_chunk:
			chunks.append(" ".join(current_chunk).strip())

		return chunks


def build_docs_from_pubmed(csv_path: Path,
						   out_path: Path
						   ) -> List[Dict[str, str]]:

	df = load_pubmed_csv(csv_path)
	docs = []
	for _, row in df.iterrows():
		chunks = smart_chunk_text(row["text"],
					 max_words = 300,
					 overlap_ratio = 0.15)
		for i, c, in enumerate(chunks):
			docs.append({
				"doc_id": str(row.get("id", "")).strip() or f"pubmed_{_}", 
				"chunk_id": f"{row.get('id', 'pubmed')}_{i}",
				"text": c,
				"title": str(row.get("title", ""))
				})

	save_json(docs, out_path)
	print(f"Saved {len(docs)} chunks to {out_path}")

	return docs


if __name__ == "__main__":
	build_docs_from_pubmed(RAW_DIR / "sample_pubmed.csv", PROC_DIR / "docs.json")