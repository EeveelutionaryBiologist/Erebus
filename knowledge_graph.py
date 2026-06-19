
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

    def retrieve_relationships(self, node_id: str, depth: int = 1) -> list[str]:
        """Returns edges within `depth` hops of `node_id` as human-readable strings."""
        if not self.G.has_node(node_id):
            return []

        subgraph = nx.ego_graph(self.G, node_id, radius=depth, undirected=False)

        facts = []
        for source, target, data in subgraph.edges(data=True):
            src_name = self.G.nodes[source].get('name', source)
            tgt_name = self.G.nodes[target].get('name', target)
            predicate = data.get('relation', 'RELATES_TO')
            facts.append(f"{src_name} [{predicate}] {tgt_name}")

        return facts

    def add_relationship(self, subject_id: str, predicate: str, object_id: str,
                         subject_name: str, object_name: str,
                         fact_ids: list[str] | None = None):
        """Adds or updates a directed edge between two entity UUID nodes."""
        self.G.add_node(subject_id, name=subject_name)
        self.G.add_node(object_id, name=object_name)

        if self.G.has_edge(subject_id, object_id):
            # Edge already exists — append new fact_ids. Predicate is left unchanged (first writer wins).
            edge_data = self.G[subject_id][object_id]
            if fact_ids:
                existing = edge_data.get('source_fact_ids', [])
                for fid in fact_ids:
                    if fid not in existing:
                        existing.append(fid)
                edge_data['source_fact_ids'] = existing
        else:
            self.G.add_edge(
                subject_id, object_id,
                relation=predicate,
                source_fact_ids=list(fact_ids) if fact_ids else []
            )

        self.write_graph()

    def rebuild_with_name_to_id_mapping(self, name_to_id: dict[str, str]) -> None:
        """Replaces string node keys with UUID keys. Used for v2 schema migration."""
        new_G = nx.DiGraph()
        for name, uid in name_to_id.items():
            new_G.add_node(uid, name=name)
        for source, target, data in self.G.edges(data=True):
            src_id = name_to_id[str(source)]
            tgt_id = name_to_id[str(target)]
            new_G.add_edge(src_id, tgt_id, **data)
        self.G = new_G
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

    def remove_relationship(self, subject_id: str, object_id: str):
        """Removes an edge by entity UUIDs and cleans up orphaned nodes."""
        if self.G.has_edge(subject_id, object_id):
            self.G.remove_edge(subject_id, object_id)
            if self.G.degree(subject_id) == 0:
                self.G.remove_node(subject_id)
            if self.G.degree(object_id) == 0:
                self.G.remove_node(object_id)
            self.write_graph()

    def write_graph(self):
        with open(self.filepath, 'w') as outfile:
            data = json_graph.node_link_data(self.G)
            json.dump(data, outfile, indent=4)
            
    def dump_all_facts(self) -> list[str]:
        """Utility: returns all graph edges as human-readable strings."""
        facts = []
        for source, target, data in self.G.edges(data=True):
            src_name = self.G.nodes[source].get('name', source)
            tgt_name = self.G.nodes[target].get('name', target)
            predicate = data.get('relation', 'RELATES_TO')
            facts.append(f"{src_name} [{predicate}] {tgt_name}")
        return facts
