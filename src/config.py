# Created by Mamoutou Fofana

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# embeddings model (sentence-transformers)
EMBED_MODEL = os.getenv("EMBED_MODEL", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

# scispaCy model name (installed via pip in requirements)
SCISPACY_MODEL = os.getenv("SCISPACY_MODEL", "en_core_sci_sm")

# FAISS / embedding files
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
EMBEDDINGS_PICKLE = MODELS_DIR / "embeddings.pkl"

# Graph
KG_PATH = MODELS_DIR / "kg_graph.gpickle"

# LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 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