# healthrag
An explainable biomedical assistant powered by Knowledge Graphs and Vector Databases.


Demo repository implementing Graph-RAG: a hybrid Retrieval-Augmented Generation system that combines vector search (embeddings + FAISS) with a knowledge graph (entity graph using networkx) to improve retrieval and context for LLMs.

This project is intended as a tutorial / reference implementation — swap in production components (Milvus, Pinecone, Neo4j, Memgraph, or OpenAI/GPT models) as needed.

## Features
- Ingest text documents and chunk them.
- Extract entities (spaCy).
- Build a simple knowledge graph (networkx) linking co-occurring entities.
- Create embeddings with `sentence-transformers` and index with FAISS.
- Hybrid retriever: vector search followed by graph traversal to expand retrieval.
- Small FastAPI service to query the pipeline.
- Scripts to run the entire pipeline.

## Quickstart (local demo)

1. Create virtualenv and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Build pipeline:
```bash
./scripts/build_all.sh
```

3. Run API:
```bash
./scripts/serve.sh
# or
uvicorn src.api.app:app --reload --port 8000
```

4. Query:
```bash
curl -s -X POST "http://localhost:8000/query" -H "Content-Type: application/json" \
  -d '{"question":"Which drug treats hypertension?","top_k":2}'
```

5. Evaluation

We provide a lightweight evaluation pipeline for both retrieval and end-to-end QA.

- Place a gold JSON in `data/gold/` with fields: `id`, `question`, `answer`, `relevant_chunk_ids`.
- Run the pipeline and index the docs (see Quickstart).
- Run evaluation:

```bash
./scripts/evaluate.sh
```