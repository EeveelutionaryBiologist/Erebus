import json
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph


class KnowledgeRelationshipGraph:
    def __init__(self, filepath):
        self.filepath = filepath
        # Load the previously stored graph as a Json
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.G = json_graph.node_link_graph(data)
        else:
            self.G = nx.DiGraph() 

    def retrieve_relationships(self, node_name: str, depth: int = 1) -> list[str]:
        """
        Yields information pieces formatted for the LLM context.
        Includes a case-insensitive fallback search.
        """
        target_node = node_name
        
        # Exact match first (fastest)
        if not self.G.has_node(target_node):
            # Fallback: case-insensitive search
            lower_name = node_name.lower()
            found = False
            for existing_node in self.G.nodes():
                if str(existing_node).lower() == lower_name:
                    target_node = existing_node
                    found = True
                    break
                    
            if not found:
                return []

        # ego_graph extracts the target node and all neighbors within the 'depth' radius
        subgraph = nx.ego_graph(self.G, target_node, radius=depth, undirected=False)
        
        extracted_facts = []
        for source, target, data in subgraph.edges(data=True):
            predicate = data.get('relation', 'RELATES_TO')
            extracted_facts.append(f"{source} [{predicate}] {target}")
            
        return extracted_facts

    def add_relationship(self, subject: str, predicate: str, object_: str, fact_ids: list[str] | None = None):
        self.G.add_node(subject)
        self.G.add_node(object_)

        if self.G.has_edge(subject, object_):
            # Edge already exists — append new fact_ids to the existing source list.
            # Predicate is left unchanged (first writer wins).
            edge_data = self.G[subject][object_]
            if fact_ids:
                existing = edge_data.get('source_fact_ids', [])
                for fid in fact_ids:
                    if fid not in existing:
                        existing.append(fid)
                edge_data['source_fact_ids'] = existing
        else:
            self.G.add_edge(
                subject, object_,
                relation=predicate,
                source_fact_ids=list(fact_ids) if fact_ids else []
            )

        self.write_graph()

    def remove_fact_reference(self, fact_id: str) -> int:
        """
        Removes fact_id from the source list of every edge that references it.
        Edges whose source_fact_ids list becomes empty are deleted, and any nodes
        that become isolated are removed. Returns the number of edges deleted.

        Legacy edges (written before source tracking was added) have no
        source_fact_ids attribute and are left untouched — they're cleaned up by
        the degree=0 orphan sweep in the consolidation routine.
        """
        for _, _, data in self.G.edges(data=True):
            source_ids = data.get('source_fact_ids')
            if source_ids and fact_id in source_ids:
                source_ids.remove(fact_id)

        edges_to_remove = [
            (s, t) for s, t, d in self.G.edges(data=True)
            if d.get('source_fact_ids') == []
        ]

        for source, target in edges_to_remove:
            self.G.remove_edge(source, target)
            if self.G.degree(source) == 0:
                self.G.remove_node(source)
            if self.G.degree(target) == 0:
                self.G.remove_node(target)

        if edges_to_remove:
            self.write_graph()

        return len(edges_to_remove)

    def remove_relationship(self, subject: str, object_: str):
        """Removes an edge, and cleans up orphaned nodes if they are left floating."""
        if self.G.has_edge(subject, object_):
            self.G.remove_edge(subject, object_)
            
            # If nodes have 0 connections left after edge removal, delete them
            if self.G.degree(subject) == 0:
                self.G.remove_node(subject)
            if self.G.degree(object_) == 0:
                self.G.remove_node(object_)
            
            self.write_graph()

    def write_graph(self):
        with open(self.filepath, 'w') as outfile:
            data = json_graph.node_link_data(self.G)
            json.dump(data, outfile, indent=4)
            
    def dump_all_facts(self) -> list[str]:
        """Utility function to view the entire knowledge base."""
        facts = []
        for source, target, data in self.G.edges(data=True):
            predicate = data.get('relation', 'RELATES_TO')
            facts.append(f"{source} [{predicate}] {target}")
        return facts
