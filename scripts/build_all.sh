#!/usr/bin/env bash
set -e
PY=python3

echo "1) Chunking PubMed sample..."
$PY - <<'PY'
from pathlib import Path
from src.ingest.chunker import build_docs_from_pubmed
from src.config import RAW_DIR, PROC_DIR
build_docs_from_pubmed(Path(RAW_DIR / "sample_pubmed.csv"), Path(PROC_DIR / "docs.json"))
PY

echo "2) Extract entities and build KG..."
$PY - <<'PY'
from pathlib import Path
import json
from src.nlp.entity_extract import BioEntityExtractor
from src.graph.builder import build_kg_from_entities, save_graph
from src.utils.input_output import load_json
docs = load_json(Path("data/processed/docs.json"))
ee = BioEntityExtractor()
entity_map = ee.extract_entities_from_docs(docs)
G = build_kg_from_entities(entity_map)
save_graph(G, Path("models/kg_graph.gpickle"))
# save edges as csv for inspection
import csv
rows=[]
for u,v,d in G.edges(data=True):
    rows.append({"u":u,"v":v,"weight":d.get("weight",1)})
with open("data/processed/kg_edges.csv","w",encoding="utf8") as f:
    writer=csv.DictWriter(f, fieldnames=["u","v","weight"])
    writer.writeheader()
    writer.writerows(rows)
print("KG built and edges saved.")
PY

echo "3) Build embeddings and index..."
$PY - <<'PY'
from pathlib import Path
from src.utils.input_output import load_json
from src.embeddings.indexer import FaissIndexer
docs = load_json(Path("data/processed/docs.json"))
idx = FaissIndexer()
idx.index_documents(docs)
idx.save_index_metadata()
print("Index built.")
PY

echo "Done"
