
import json
from collections import deque
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph


class KnowledgeRelationshipGraph:
    """
    Directed multi-graph of entity relationships backed by a JSON file.

    Node keys are entity UUIDs; human-readable names are stored as node attributes.
    Each edge carries a 'relation' string and a 'source_fact_ids' list so that
    fact deletions can propagate without a full edge scan.

    _fact_edge_index is a derived, in-memory acceleration structure:
        fact_id → [(subject_id, object_id, edge_key), ...]
    Invariant: it must always equal the union of source_fact_ids across all edges.
    Every method that mutates edge source lists must keep the index in sync.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.G = json_graph.node_link_graph(data)
            # Transparently upgrade legacy DiGraph saves to MultiDiGraph.
            if not isinstance(self.G, nx.MultiDiGraph):
                self.G = nx.MultiDiGraph(self.G)
        else:
            self.G = nx.MultiDiGraph()
        self._fact_edge_index: dict[str, list[tuple[str, str, int]]] = self._build_fact_edge_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_fact_edge_index(self) -> dict[str, list[tuple[str, str, int]]]:
        index: dict[str, list[tuple[str, str, int]]] = {}
        for s, t, k, data in self.G.edges(data=True, keys=True):
            for fid in data.get('source_fact_ids', []):
                index.setdefault(fid, []).append((s, t, k))
        return index

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_relationships(self, node_id: str, depth: int = 1) -> list[str]:
        """Returns all edges within `depth` hops of `node_id` as human-readable strings."""
        if not self.G.has_node(node_id):
            return []

        subgraph = nx.ego_graph(self.G, node_id, radius=depth, undirected=True)

        facts = []
        for source, target, data in subgraph.edges(data=True):
            src_name = self.G.nodes[source].get('name', source)
            tgt_name = self.G.nodes[target].get('name', target)
            predicate = data.get('relation', 'RELATES_TO')
            facts.append(f"{src_name} [{predicate}] {tgt_name}")

        return facts

    def dump_all_facts(self) -> list[str]:
        """Returns all graph edges as human-readable strings."""
        facts = []
        for source, target, data in self.G.edges(data=True):
            src_name = self.G.nodes[source].get('name', source)
            tgt_name = self.G.nodes[target].get('name', target)
            predicate = data.get('relation', 'RELATES_TO')
            facts.append(f"{src_name} [{predicate}] {tgt_name}")
        return facts

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_relationship(self, subject_id: str, predicate: str, object_id: str,
                         subject_name: str, object_name: str,
                         fact_ids: list[str] | None = None,
                         persist: bool = True):
        """Adds or updates a directed edge between two entity UUID nodes.

        If an edge with the same predicate already exists between this pair, the
        new fact_ids are appended to it (preserving the existing relationship).
        A genuinely different predicate gets its own edge (MultiDiGraph behaviour).

        Set persist=False when calling in a tight loop and flush with write_graph()
        once at the end to avoid serialising the full graph on every triple.
        """
        self.G.add_node(subject_id, name=subject_name)
        self.G.add_node(object_id, name=object_name)

        # Look for an existing edge with the same predicate.
        existing_key = None
        if self.G.has_edge(subject_id, object_id):
            for k, data in self.G[subject_id][object_id].items():
                if data.get('relation') == predicate:
                    existing_key = k
                    break

        if existing_key is not None:
            edge_data = self.G[subject_id][object_id][existing_key]
            if fact_ids:
                existing = edge_data.setdefault('source_fact_ids', [])
                for fid in fact_ids:
                    if fid not in existing:
                        existing.append(fid)
                        self._fact_edge_index.setdefault(fid, []).append(
                            (subject_id, object_id, existing_key)
                        )
        else:
            new_key = self.G.add_edge(
                subject_id, object_id,
                relation=predicate,
                source_fact_ids=list(fact_ids) if fact_ids else []
            )
            if fact_ids:
                for fid in fact_ids:
                    self._fact_edge_index.setdefault(fid, []).append(
                        (subject_id, object_id, new_key)
                    )

        if persist:
            self.write_graph()

    def rebuild_with_name_to_id_mapping(self, name_to_id: dict[str, str]) -> None:
        """Replaces string node keys with UUID keys. Used for the v2 schema migration."""
        new_G = nx.MultiDiGraph()
        for name, uid in name_to_id.items():
            new_G.add_node(uid, name=name)
        for source, target, data in self.G.edges(data=True):
            src_id = name_to_id[str(source)]
            tgt_id = name_to_id[str(target)]
            new_G.add_edge(src_id, tgt_id, **data)
        self.G = new_G
        self._fact_edge_index = self._build_fact_edge_index()
        self.write_graph()

    def remove_fact_reference(self, fact_id: str) -> int:
        """Removes fact_id from every edge that references it.

        Edges whose source_fact_ids list becomes empty are deleted.
        Nodes that become isolated after edge deletion are removed.
        Returns the number of edges deleted.

        O(1) in the number of edges, not O(E), courtesy of _fact_edge_index.
        Legacy edges with no source_fact_ids are left in place and cleaned up
        by the degree-0 orphan sweep in the consolidation routine.
        """
        edge_refs = self._fact_edge_index.pop(fact_id, [])
        if not edge_refs:
            return 0

        edges_to_remove: list[tuple[str, str, int]] = []
        for s, t, k in edge_refs:
            if not self.G.has_edge(s, t, key=k):
                continue
            source_ids: list = self.G[s][t][k].get('source_fact_ids', [])
            if fact_id in source_ids:
                source_ids.remove(fact_id)
            if not source_ids:
                edges_to_remove.append((s, t, k))

        for s, t, k in edges_to_remove:
            if self.G.has_edge(s, t, key=k):
                self.G.remove_edge(s, t, key=k)
            if self.G.has_node(s) and self.G.degree(s) == 0:
                self.G.remove_node(s)
            if self.G.has_node(t) and self.G.degree(t) == 0:
                self.G.remove_node(t)

        if edges_to_remove:
            self.write_graph()

        return len(edges_to_remove)

    def remove_relationship(self, subject_id: str, object_id: str):
        """Removes all edges between subject_id and object_id and cleans up orphaned nodes."""
        if not self.G.has_edge(subject_id, object_id):
            return

        for k, data in dict(self.G[subject_id][object_id]).items():
            for fid in data.get('source_fact_ids', []):
                if fid in self._fact_edge_index:
                    self._fact_edge_index[fid] = [
                        (s, t, ki) for s, t, ki in self._fact_edge_index[fid]
                        if not (s == subject_id and t == object_id and ki == k)
                    ]
                    if not self._fact_edge_index[fid]:
                        del self._fact_edge_index[fid]
            if self.G.has_edge(subject_id, object_id, key=k):
                self.G.remove_edge(subject_id, object_id, key=k)

        if self.G.has_node(subject_id) and self.G.degree(subject_id) == 0:
            self.G.remove_node(subject_id)
        if self.G.has_node(object_id) and self.G.degree(object_id) == 0:
            self.G.remove_node(object_id)
        self.write_graph()

    def remove_fact_node(self, node_id: str) -> None:
        """Remove node_id as a graph node, cleaning _fact_edge_index for all incident edges.

        Complements remove_fact_reference(): that method removes a fact from edge
        source_fact_ids lists; this method removes a fact that *is itself a node*
        (e.g. a current-state fact in the temporal graph). Call both when a fact that
        may occupy either role is deleted.
        """
        if not self.G.has_node(node_id):
            return
        all_incident = (
            list(self.G.out_edges(node_id, data=True, keys=True))
            + list(self.G.in_edges(node_id, data=True, keys=True))
        )
        for subj, obj, key, data in all_incident:
            for fid in list(data.get("source_fact_ids", [])):
                entries = self._fact_edge_index.get(fid, [])
                updated = [e for e in entries if e != (subj, obj, key)]
                if updated:
                    self._fact_edge_index[fid] = updated
                else:
                    self._fact_edge_index.pop(fid, None)
        self.G.remove_node(node_id)
        self.write_graph()

    def remove_entity_node(self, node_id: str, persist: bool = True) -> None:
        """Remove an entity node, cleaning _fact_edge_index for all its incident edges.

        Mirrors remove_fact_node() but operates on entity-layer nodes (knowledge_graph)
        rather than temporal-state nodes (temporal_graph). Use persist=False inside a
        batch loop and call write_graph() once at the end.
        """
        if not self.G.has_node(node_id):
            return
        all_incident = (
            list(self.G.out_edges(node_id, data=True, keys=True))
            + list(self.G.in_edges(node_id, data=True, keys=True))
        )
        for subj, obj, key, data in all_incident:
            for fid in list(data.get("source_fact_ids", [])):
                entries = self._fact_edge_index.get(fid, [])
                updated = [e for e in entries if e != (subj, obj, key)]
                if updated:
                    self._fact_edge_index[fid] = updated
                else:
                    self._fact_edge_index.pop(fid, None)
        self.G.remove_node(node_id)
        if persist:
            self.write_graph()

    def retrieve_predecessor_chain(self, fact_id: str, max_depth: int = 10) -> list[dict]:
        """BFS traversal following only PRECEDED_BY edges from fact_id.

        Returns list of {fact_id: str, depth: int} ordered by hop distance.
        CONCURRENT_WITH edges are never followed — use get_concurrent_with() per hop.
        """
        if not self.G.has_node(fact_id):
            return []
        result: list[dict] = []
        queue: deque[tuple[str, int]] = deque([(fact_id, 0)])
        visited: set[str] = {fact_id}
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for _, succ, edge_data in self.G.out_edges(node, data=True):
                if edge_data.get("relation") == "PRECEDED_BY" and succ not in visited:
                    visited.add(succ)
                    result.append({"fact_id": succ, "depth": depth + 1})
                    queue.append((succ, depth + 1))
        return result

    def get_concurrent_with(self, fact_id: str) -> list[str]:
        """Returns fact_ids of all CONCURRENT_WITH neighbors of fact_id."""
        if not self.G.has_node(fact_id):
            return []
        return [
            succ
            for _, succ, edge_data in self.G.out_edges(fact_id, data=True)
            if edge_data.get("relation") == "CONCURRENT_WITH"
        ]

    def clear(self):
        """Wipes all nodes, edges, and the fact index, then persists the empty graph."""
        self.G.clear()
        self._fact_edge_index = {}
        self.write_graph()

    def write_graph(self):
        with open(self.filepath, 'w') as outfile:
            data = json_graph.node_link_data(self.G)
            json.dump(data, outfile, indent=4)
