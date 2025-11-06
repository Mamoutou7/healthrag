# -*- coding: utf-8 -*-
"""
Knowledge Graph Construction from Entities

Created by Mamoutou Fofana
Date: 2025-10-25
"""

import logging
import pickle
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import networkx as nx

from ..config import KG_PATH, PROC_DIR
from ..utils.input_output import save_pickle, save_json

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



class KnowledgeGraphBuilder:
    """
    Build and manage a co-occurrence Knowledge Graph (KG) from text entities.

    Nodes represent unique entity mentions (normalized text + label).
    Edges represent co-occurrences within the same text chunk.
    """
    def __init__(self, entity_map: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize the KnowledgeGraphBuilder.

        Args:
            entity_map: Mapping of chunk IDs to lists of entity dictionaries.
                        Each entity dictionary must include a 'text' key
                        and may optionally include a 'label'.
        """

        if not isinstance(entity_map, dict):
            raise TypeError("`entity_map` must be a dictionary.")

        self.entity_map = entity_map
        self.graph = nx.Graph()

    def build_kg_graph(self) -> nx.Graph:
        """
        Construct the knowledge graph from the provided entity map.

        Returns:
            nx.Graph: A NetworkX graph with nodes (entities) and edges (co-occurrences).
        """
        self._validate_input()
        nodes_data = self.extract_nodes()
        edge_data = self._extract_edges()

        self._add_nodes(nodes_data)
        self._add_edges(edge_data)
        self._finalize_graph()

        logger.info(
            "Knowledge graph built successfully: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return self.graph

    def save_kg_graph(
            self,
            graph_path: Optional[Path] = None
    ) -> None:
        """
        Save the built knowledge graph to a pickle file.
        Args:
            graph_path: Optional path to save the graph (defaults to KG_PATH).
        """

        if not isinstance(self.graph, nx.Graph):
            raise ValueError("Graph not built. Call `build_kg_graph()` before saving.")

        graph_path = graph_path or KG_PATH
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        with open(graph_path, "wb") as file:
            pickle.dump(self.graph, file, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("Knowledge graph saved to %s", graph_path)

    # Private helper methods
    def _validate_input(self) -> None:
        """Validate the structure of the input entity map."""
        if not self.entity_map:
            logger.warning("Empty entity map provided. Returning empty graph.")
            return

        for chunk_id, entities in self.entity_map.items():
            if not isinstance(entities, list):
                raise ValueError(f"Chunk {chunk_id} must map to a list of entities.")
            for entity in entities:
                if not isinstance(entity, dict) or "text" not in entity:
                    raise ValueError(
                        f"Invalid entity structure in chunk {chunk_id}. "
                        "Each entity must be a dict containing a 'text' key."
                    )

    def extract_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Extract normalized entity nodes from the entity map."""
        node_data: Dict[str, Dict[str, Any]] = {}
        for chunk_id, entities in self.entity_map.items():
            for entity in entities:
                name = entity["text"].strip().lower()
                if not name:
                    continue
                label = entity.get("label", "")
                node_info = node_data.setdefault(name, {"label": label, "chunks": set()})
                node_info["chunks"].add(chunk_id)

        return node_data


    def _extract_edges(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Extract co-occurrence edges from entity mentions.
        """
        edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for chunk_id, entities in self.entity_map.items():
            names = [e["text"].strip().lower() for e in entities if e["text"].strip()]
            for a, b in combinations(sorted(set(names)), 2):
                edge = (a, b)
                edge_info = edge_data.setdefault(edge, {"weight": 0, "chunks": set()})
                edge_info["weight"] += 1
                edge_info["chunks"].add(chunk_id)

        return edge_data


    def _add_nodes(
            self,
            node_data: Dict[str, Dict[str, Any]]
    ) -> None:
        """Add nodes to the NetworkX graph."""
        self.graph.add_nodes_from(
            (name, {"label": data["label"], "chunks": data["chunks"]})
            for name, data in node_data.items()
        )


    def _add_edges(
            self,
            edge_data: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> None:
        """Add edges to the NetworkX graph."""
        self.graph.add_edges_from(
            (u, v, {"weight": data["weight"], "chunks": data["chunks"]})
            for (u, v), data in edge_data.items()
        )

    def _finalize_graph(self) -> None:
        """Convert sets to lists for serialization compatibility."""
        for _, data in self.graph.nodes(data=True):
            data["chunks"] = list(data["chunks"])
        for _, _, data in self.graph.edges(data=True):
            data["chunks"] = list(data["chunks"])

if __name__ == "__main__":
    # Example usage with sample data
    sample_entities = {
        "chunk1": [
            {"text": "Remdesivir", "label": "ENTITY", "start": 0, "end": 10},
            {"text": "COVID-19", "label": "ENTITY", "start": 35, "end": 43}
        ],
        "chunk2": [
            {"text": "Remdesivir", "label": "ENTITY", "start": 0, "end": 10},
            {"text": "severe", "label": "ENTITY", "start": 28, "end": 34}
        ]
    }

    try:
        builder = KnowledgeGraphBuilder(sample_entities)
        kg = builder.build_kg_graph()
        logger.info(
            "Graph summary — Nodes: %d | Edges: %d | Density: %.4f",
            kg.number_of_nodes(),
            kg.number_of_edges(),
            nx.density(kg),
        )

        builder.save_kg_graph(Path(PROC_DIR) / "sample_kg.pkl")
    except Exception as exc:
        logger.error("Error while building or saving the graph: %s", exc, exc_info=True)
