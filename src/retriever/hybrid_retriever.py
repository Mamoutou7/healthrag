# Created by Mamoutou Fofana
# Date: 10/25/2025

from ..embeddings.indexer import FaissIndexer
from ..nlp.entity_extract import BioEntityExtractor
from ..config import KG_PATH
from pathlib import Path
import networkx as nx
import pickle
from collections import defaultdict


class HealthHybridRetriever:
    def __init__(
        self,
        indexer: FaissIndexer,
        ee: BioEntityExtractor,
        kg_path: Path = Path(KG_PATH),
    ):
        self.faiss = indexer
        self.ee = ee
        self.graph = pickle.load(
            open(kg_path, "rb")) if kg_path.exists() else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        graph_hops: int = 1,
        alpha: float = 0.5
    ):
        """
        Hybrid retrieval combining FAISS vector search and KG expansion.

        Args:
            query: User query string.
            top_k: Number of top vector results to retrieve.
            graph_hops: Depth for KG expansion around query entities.
            alpha: Weight for combining FAISS scores and KG hits
            (0=only FAISS, 1=only KG).

        Returns:
            Dict with combined results:
                - vector_hits: raw FAISS hits
                - expanded_chunk_ids: KG-expanded chunk IDs
                - matched_nodes: entities matched in the KG
                - context: concatenated text from top chunks
                - hybrid_scores: dictionary of chunk_id -> combined score
        """
        # Vector search
        vector_hits = self.faiss.search(query, top_k=top_k)
        chunk_ids = [h["chunk_id"] for h in vector_hits]
        context_chunks = [h["text"] for h in vector_hits]
        vector_scores = {h["chunk_id"]: h["score"] for h in vector_hits}

        # Extract entities from top chunks + query
        ents_from_chunks = {
            cid: self.ee.extract(text) for cid, text in zip(
                chunk_ids,
                context_chunks
                )
        }
        query_ents = self.ee.extract(query)

        # KG-based expansion
        expanded_chunks = set()
        matched_nodes = []
        if self.graph:
            # Combine query entities + chunk entities
            all_ents = [e["text"] for e in query_ents]
            for ents in ents_from_chunks.values():
                all_ents.extend([e["text"] for e in ents])

            for ent_text in all_ents:
                if ent_text in self.graph.nodes:
                    matched_nodes.append(ent_text)
                    # get neighbors up to graph_hops
                    neighbors = nx.single_source_shortest_path_length(
                        self.graph, ent_text, cutoff=graph_hops
                    )
                    for n in neighbors:
                        expanded_chunks.update(
                            self.graph.nodes[n].get("chunks", []))

        # Combine FAISS and KG hits with scores
        hybrid_scores = defaultdict(float)
        for cid in chunk_ids:
            hybrid_scores[cid] += (1 - alpha) * vector_scores.get(cid, 0.0)
        for cid in expanded_chunks:
            hybrid_scores[cid] += alpha  # assign uniform KG boost

        # Sort chunks by hybrid score
        sorted_chunks = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        hybrid_chunk_ids = [cid for cid, _ in sorted_chunks]

        return {
            "vector_hits": vector_hits,
            "expanded_chunk_ids": list(expanded_chunks),
            "matched_nodes": matched_nodes,
            "context": " ".join(context_chunks),
            "hybrid_scores": hybrid_scores,
            "sorted_chunk_ids": hybrid_chunk_ids,
        }
