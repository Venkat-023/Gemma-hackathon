from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import networkx as nx


NODE_COLORS = {
    "PAPER": "#4A90E2",
    "DISEASE": "#E74C3C",
    "PROTEIN": "#2ECC71",
    "GENE": "#9B59B6",
    "CHEMICAL": "#F39C12",
    "METHOD": "#1ABC9C",
    "CONCEPT": "#7F8C8D",
    "CLUSTER": "#F59E0B",
}
NODE_TYPES = {"PAPER", "DISEASE", "PROTEIN", "GENE", "CHEMICAL", "METHOD", "CONCEPT", "CLUSTER"}
BRIDGE_MIN_PAPERS = 2


class FastGraph:
    def __init__(self, graph_path: str = "./data/fast_knowledge_graph.json") -> None:
        self.graph_path = Path(graph_path)
        self.graph = nx.MultiDiGraph()

    def add_paper(self, paper_id: int, title: str, entities: dict[str, list[str]], relationships: list[dict]) -> dict:
        paper_node = f"P:{paper_id}"
        self.graph.add_node(
            paper_node,
            label=title[:80],
            type="PAPER",
            node_type="PAPER",
            paper_id=paper_id,
            size=40,
            paper_ids=[paper_id],
        )
        added_entities = []

        for entity_type, names in entities.items():
            normalized_type = entity_type.upper() if entity_type.upper() in NODE_TYPES else "CONCEPT"
            for name in names or []:
                label = str(name).strip()
                if len(label) < 2:
                    continue
                entity_node = self._entity_id(normalized_type, label)
                if not self.graph.has_node(entity_node):
                    self.graph.add_node(
                        entity_node,
                        label=label,
                        type=normalized_type,
                        node_type=normalized_type,
                        name_key=label.lower(),
                        papers=[],
                        paper_ids=[],
                        mention_count=0,
                        is_bridge=False,
                        size=20,
                    )
                node_data = self.graph.nodes[entity_node]
                papers = set(node_data.get("papers", []))
                papers.add(paper_id)
                node_data["papers"] = sorted(papers)
                node_data["paper_ids"] = sorted(papers)
                node_data["mention_count"] = int(node_data.get("mention_count", 0)) + 1
                node_data["is_bridge"] = len(papers) >= BRIDGE_MIN_PAPERS
                node_data["size"] = min(34, 16 + node_data["mention_count"] * 2 + len(papers) * 3)
                self.graph.add_edge(
                    paper_node,
                    entity_node,
                    relation="MENTIONS",
                    paper_id=paper_id,
                    weight=1.0,
                )
                added_entities.append(entity_node)

        added_relationships = 0
        for relationship in relationships or []:
            source = str(relationship.get("source", "")).strip()
            target = str(relationship.get("target", "")).strip()
            if not source or not target:
                continue
            source_node = self._find_or_create_concept(source, paper_id)
            target_node = self._find_or_create_concept(target, paper_id)
            self.graph.add_edge(
                source_node,
                target_node,
                relation=relationship.get("relation", "relates_to"),
                confidence=float(relationship.get("confidence", 0.7)),
                weight=float(relationship.get("confidence", 0.7)),
                paper_id=paper_id,
                evidence=relationship.get("evidence", ""),
            )
            added_relationships += 1

        self._add_bridge_edges()
        self.save()
        return {
            "entity_nodes": len(set(added_entities)),
            "relationship_edges": added_relationships,
            "bridge_concepts": len(self.find_cross_paper_bridges(top_n=999)),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }

    def to_frontend_json(self, paper_id: int | None = None, max_nodes: int = 120) -> dict[str, Any]:
        export = self.get_paper_neighbourhood(paper_id, depth=2) if paper_id else self.get_full_export(max_nodes=max_nodes)
        nodes = [
            {
                "data": {
                    "id": node["id"],
                    "label": node["label"],
                    "type": node["type"],
                    "color": NODE_COLORS.get(node["type"], "#95A5A6"),
                    "size": node["size"],
                    "papers": node.get("paper_ids", []),
                    "is_bridge": node.get("is_bridge", False),
                    "pagerank": node.get("pagerank", 0.0),
                }
            }
            for node in export["nodes"]
        ]
        edges = [
            {
                "data": {
                    "source": edge["source"],
                    "target": edge["target"],
                    "relation": edge["relation"],
                    "weight": edge.get("weight", 1.0),
                    "paper_id": edge.get("paper_id"),
                    "is_bridge": edge.get("is_bridge", False),
                }
            }
            for edge in export["edges"]
        ]
        return {"elements": {"nodes": nodes, "edges": edges}, "stats": export.get("stats", {})}

    def get_full_export(self, max_nodes: int = 500) -> dict[str, Any]:
        self._write_centrality()
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type", attrs.get("type", "CONCEPT"))
            nodes.append(
                {
                    "id": node_id,
                    "label": attrs.get("label", node_id)[:60],
                    "type": node_type,
                    "mention_count": attrs.get("mention_count", 1),
                    "paper_count": len(attrs.get("paper_ids", attrs.get("papers", []))),
                    "paper_ids": attrs.get("paper_ids", attrs.get("papers", [])),
                    "is_bridge": attrs.get("is_bridge", False),
                    "pagerank": attrs.get("pagerank", 0.0),
                    "betweenness": attrs.get("betweenness", 0.0),
                    "size": attrs.get("size", 20),
                }
            )
        nodes.sort(key=lambda node: node["pagerank"] + node["paper_count"] * 0.01, reverse=True)
        nodes = nodes[:max_nodes]
        node_set = {node["id"] for node in nodes}

        edges = []
        seen = set()
        for source, target, attrs in self.graph.edges(data=True):
            if source not in node_set or target not in node_set:
                continue
            key = (source, target, attrs.get("relation", ""))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation", "related"),
                    "weight": attrs.get("weight", attrs.get("confidence", 1.0)),
                    "paper_id": attrs.get("paper_id"),
                    "is_bridge": attrs.get("relation") == "BRIDGES",
                }
            )
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "bridge_concepts": len(self.find_cross_paper_bridges(top_n=999)),
                "total_papers": len([n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "PAPER"]),
            },
        }

    def get_paper_neighbourhood(self, paper_id: int | None, depth: int = 2) -> dict[str, Any]:
        if paper_id is None:
            return self.get_full_export()
        paper_node = f"P:{paper_id}"
        if not self.graph.has_node(paper_node):
            return {"nodes": [], "edges": [], "stats": {}}
        undirected = self.graph.to_undirected()
        selected = set(nx.single_source_shortest_path_length(undirected, paper_node, cutoff=depth).keys())
        sub = self.graph.subgraph(selected).copy()
        return self._export_subgraph(sub)

    def find_cross_paper_bridges(self, top_n: int = 20) -> list[dict[str, Any]]:
        bridges = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") in {"PAPER", "CLUSTER"}:
                continue
            paper_ids = attrs.get("paper_ids", attrs.get("papers", []))
            if len(paper_ids) < BRIDGE_MIN_PAPERS:
                continue
            paper_pairs = []
            for i, paper_a in enumerate(paper_ids):
                for paper_b in paper_ids[i + 1 :]:
                    paper_pairs.append(
                        {
                            "paper_a": paper_a,
                            "title_a": self._paper_title(paper_a),
                            "paper_b": paper_b,
                            "title_b": self._paper_title(paper_b),
                            "relationship": "shared_concept",
                        }
                    )
            bridges.append(
                {
                    "concept": attrs.get("label", node_id),
                    "concept_id": node_id,
                    "concept_type": attrs.get("node_type", "CONCEPT"),
                    "paper_count": len(paper_ids),
                    "mention_count": attrs.get("mention_count", 1),
                    "paper_pairs": paper_pairs[:8],
                    "importance_score": round(attrs.get("mention_count", 1) * math.log(len(paper_ids) + 1), 3),
                }
            )
        return sorted(bridges, key=lambda item: item["importance_score"], reverse=True)[:top_n]

    def get_concept_clusters(self) -> list[dict[str, Any]]:
        entity_nodes = [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") not in {"PAPER", "CLUSTER"}
        ]
        if len(entity_nodes) < 2:
            return []
        subgraph = self.graph.subgraph(entity_nodes).to_undirected()
        if subgraph.number_of_edges() == 0:
            communities = list(nx.connected_components(subgraph))
        else:
            communities = list(nx.community.greedy_modularity_communities(subgraph, weight="weight"))
        clusters = []
        for index, community in enumerate(communities):
            if len(community) < 2:
                continue
            members = list(community)
            papers = set()
            for member in members:
                papers.update(self.graph.nodes[member].get("paper_ids", []))
            ranked = sorted(
                members,
                key=lambda node: self.graph.nodes[node].get("mention_count", 1)
                * max(1, len(self.graph.nodes[node].get("paper_ids", []))),
                reverse=True,
            )
            labels = [self.graph.nodes[node].get("label", node) for node in ranked[:8]]
            clusters.append(
                {
                    "cluster_id": f"CLUSTER:{index:03d}",
                    "centroid_label": labels[0],
                    "size": len(members),
                    "top_concepts": labels,
                    "paper_ids": sorted(papers),
                    "paper_count": len(papers),
                }
            )
        return sorted(clusters, key=lambda cluster: cluster["size"] * max(1, cluster["paper_count"]), reverse=True)

    def find_research_gaps(self, top_n: int = 10) -> list[dict[str, Any]]:
        paper_to_entities: dict[Any, set[str]] = defaultdict(set)
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") in {"PAPER", "CLUSTER"}:
                continue
            for paper_id in attrs.get("paper_ids", []):
                paper_to_entities[paper_id].add(node_id)

        candidates = set()
        for entities in paper_to_entities.values():
            ordered = sorted(entities)
            for i, concept_a in enumerate(ordered):
                for concept_b in ordered[i + 1 :]:
                    candidates.add((concept_a, concept_b))

        undirected = self.graph.to_undirected()
        gaps = []
        for concept_a, concept_b in candidates:
            if undirected.has_edge(concept_a, concept_b):
                continue
            papers_a = set(self.graph.nodes[concept_a].get("paper_ids", []))
            papers_b = set(self.graph.nodes[concept_b].get("paper_ids", []))
            shared = papers_a & papers_b
            if not shared:
                continue
            neighbors_a = set(undirected.neighbors(concept_a)) if concept_a in undirected else set()
            neighbors_b = set(undirected.neighbors(concept_b)) if concept_b in undirected else set()
            union = neighbors_a | neighbors_b
            overlap = len(neighbors_a & neighbors_b) / len(union) if union else 0.0
            score = len(shared) * (1.0 - overlap)
            gaps.append(
                {
                    "concept_a": self.graph.nodes[concept_a].get("label", concept_a),
                    "concept_b": self.graph.nodes[concept_b].get("label", concept_b),
                    "concept_a_id": concept_a,
                    "concept_b_id": concept_b,
                    "shared_paper_count": len(shared),
                    "shared_paper_ids": sorted(shared),
                    "jaccard_neighbour_overlap": round(overlap, 4),
                    "gap_score": round(score, 4),
                    "gap_type": "unexplored",
                }
            )
        return sorted(gaps, key=lambda gap: gap["gap_score"], reverse=True)[:top_n]

    def get_hub_concepts(self, top_n: int = 15) -> list[dict[str, Any]]:
        centrality = self._centrality()
        hubs = []
        for node_id, scores in centrality.items():
            attrs = self.graph.nodes[node_id]
            paper_count = len(attrs.get("paper_ids", []))
            hubs.append(
                {
                    "node_id": node_id,
                    "label": attrs.get("label", node_id),
                    "type": attrs.get("node_type", attrs.get("type", "CONCEPT")),
                    "paper_count": paper_count,
                    "mention_count": attrs.get("mention_count", 1),
                    "pagerank": scores["pagerank"],
                    "betweenness": scores["betweenness"],
                    "hub_score": round(scores["pagerank"] * (1 + paper_count), 8),
                }
            )
        return sorted(hubs, key=lambda hub: hub["hub_score"], reverse=True)[:top_n]

    def build_hypothesis_context(self, query: str, vector_results: list[dict] | None = None) -> str:
        parts = []
        if vector_results:
            parts.append("## Retrieved Evidence")
            for result in vector_results[:6]:
                meta = result.get("metadata", {})
                parts.append(f"[{meta.get('title', '?')} | {meta.get('section', '?')}] {result.get('text', '')[:320]}")
        bridges = self.find_cross_paper_bridges(top_n=8)
        if bridges:
            parts.append("\n## Cross-Paper Concept Bridges")
            for bridge in bridges[:5]:
                parts.append(
                    f"{bridge['concept']} appears in {bridge['paper_count']} papers "
                    f"with importance {bridge['importance_score']}."
                )
        gaps = self.find_research_gaps(top_n=5)
        if gaps:
            parts.append("\n## Unexplored Concept Gaps")
            for gap in gaps:
                parts.append(
                    f"{gap['concept_a']} <-> {gap['concept_b']} shared in "
                    f"{gap['shared_paper_count']} papers but lacks a direct relationship."
                )
        hubs = self.get_hub_concepts(top_n=6)
        if hubs:
            parts.append("\n## Hub Concepts")
            parts.append(", ".join(f"{hub['label']} ({hub['paper_count']} papers)" for hub in hubs))
        return "\n".join(parts)

    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        self.graph_path.write_text(__import__("json").dumps(data, separators=(",", ":")), encoding="utf-8")

    def _find_or_create_concept(self, label: str, paper_id: int) -> str:
        label = label.strip()
        lowered = label.lower()
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == "PAPER":
                continue
            if attrs.get("name_key") == lowered:
                papers = set(attrs.get("paper_ids", []))
                papers.add(paper_id)
                attrs["paper_ids"] = sorted(papers)
                attrs["papers"] = sorted(papers)
                attrs["is_bridge"] = len(papers) >= BRIDGE_MIN_PAPERS
                return node_id
        node_id = self._entity_id("CONCEPT", label)
        self.graph.add_node(
            node_id,
            label=label,
            type="CONCEPT",
            node_type="CONCEPT",
            name_key=lowered,
            papers=[paper_id],
            paper_ids=[paper_id],
            mention_count=1,
            is_bridge=False,
            size=20,
        )
        return node_id

    def _add_bridge_edges(self) -> None:
        for node_id, attrs in list(self.graph.nodes(data=True)):
            if attrs.get("node_type") in {"PAPER", "CLUSTER"}:
                continue
            paper_ids = attrs.get("paper_ids", [])
            if len(paper_ids) < BRIDGE_MIN_PAPERS:
                continue
            for paper_id in paper_ids:
                paper_node = f"P:{paper_id}"
                if self.graph.has_node(paper_node):
                    self.graph.add_edge(
                        node_id,
                        paper_node,
                        relation="BRIDGES",
                        weight=len(paper_ids),
                        paper_id=paper_id,
                    )

    def _centrality(self) -> dict[str, dict[str, float]]:
        entity_nodes = [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("node_type") not in {"PAPER", "CLUSTER"}
        ]
        if not entity_nodes:
            return {}
        subgraph = self.graph.subgraph(entity_nodes).to_undirected()
        if subgraph.number_of_nodes() <= 1:
            return {node_id: {"pagerank": 0.0, "betweenness": 0.0} for node_id in entity_nodes}
        pagerank = nx.pagerank(subgraph, weight="weight", alpha=0.85)
        betweenness = nx.betweenness_centrality(subgraph, weight="weight")
        return {
            node_id: {
                "pagerank": round(pagerank.get(node_id, 0.0), 6),
                "betweenness": round(betweenness.get(node_id, 0.0), 6),
            }
            for node_id in entity_nodes
        }

    def _write_centrality(self) -> None:
        for node_id, scores in self._centrality().items():
            self.graph.nodes[node_id]["pagerank"] = scores["pagerank"]
            self.graph.nodes[node_id]["betweenness"] = scores["betweenness"]

    def _export_subgraph(self, subgraph: nx.MultiDiGraph) -> dict[str, Any]:
        nodes = []
        for node_id, attrs in subgraph.nodes(data=True):
            node_type = attrs.get("node_type", attrs.get("type", "CONCEPT"))
            nodes.append(
                {
                    "id": node_id,
                    "label": attrs.get("label", node_id)[:60],
                    "type": node_type,
                    "mention_count": attrs.get("mention_count", 1),
                    "paper_count": len(attrs.get("paper_ids", attrs.get("papers", []))),
                    "paper_ids": attrs.get("paper_ids", attrs.get("papers", [])),
                    "is_bridge": attrs.get("is_bridge", False),
                    "pagerank": attrs.get("pagerank", 0.0),
                    "betweenness": attrs.get("betweenness", 0.0),
                    "size": attrs.get("size", 20),
                }
            )
        edges = [
            {
                "source": source,
                "target": target,
                "relation": attrs.get("relation", "related"),
                "weight": attrs.get("weight", attrs.get("confidence", 1.0)),
                "paper_id": attrs.get("paper_id"),
                "is_bridge": attrs.get("relation") == "BRIDGES",
            }
            for source, target, attrs in subgraph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges, "stats": {"total_nodes": len(nodes), "total_edges": len(edges)}}

    def _paper_title(self, paper_id: int) -> str:
        node_id = f"P:{paper_id}"
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id].get("label", f"Paper {paper_id}")
        return f"Paper {paper_id}"

    def _entity_id(self, entity_type: str, label: str) -> str:
        safe = " ".join(label.lower().split())
        return f"E:{entity_type}:{safe}"
