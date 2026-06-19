"""
Unit tests for KnowledgeRelationshipGraph.

All tests are fully isolated (no server, no GGUF models).
Each test gets a fresh tmp graph via the `fresh_graph` fixture.
"""

import json
import networkx as nx
import pytest

from knowledge_graph import KnowledgeRelationshipGraph


# ---------------------------------------------------------------------------
# MultiDiGraph: multi-predicate support
# ---------------------------------------------------------------------------

def test_two_predicates_same_pair_both_survive(fresh_graph):
    """Two distinct predicates between the same entity pair must create two edges."""
    fresh_graph.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    fresh_graph.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f2"])

    edges = list(fresh_graph.G.edges(data=True))
    assert len(edges) == 2
    predicates = {d["relation"] for _, _, d in edges}
    assert predicates == {"WORKS_AT", "OWNS"}


def test_same_predicate_appends_fact_ids_not_new_edge(fresh_graph):
    """Re-adding the same predicate must append fact_ids to the existing edge, not fork it."""
    fresh_graph.add_relationship("A", "LIKES", "C", "Alice", "Coffee", fact_ids=["f1"])
    fresh_graph.add_relationship("A", "LIKES", "C", "Alice", "Coffee", fact_ids=["f2"])

    edges = list(fresh_graph.G.edges(data=True))
    assert len(edges) == 1
    assert set(edges[0][2]["source_fact_ids"]) == {"f1", "f2"}


def test_retrieve_relationships_returns_all_predicates(fresh_graph):
    """retrieve_relationships must surface all predicates between a multi-edge pair."""
    fresh_graph.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    fresh_graph.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f2"])

    facts = fresh_graph.retrieve_relationships("A", depth=1)
    assert len(facts) == 2
    assert any("WORKS_AT" in f for f in facts)
    assert any("OWNS" in f for f in facts)


def test_retrieve_relationships_unknown_node_returns_empty(fresh_graph):
    assert fresh_graph.retrieve_relationships("nonexistent") == []


# ---------------------------------------------------------------------------
# Fact→edge index correctness
# ---------------------------------------------------------------------------

def test_index_populated_after_add(fresh_graph):
    fresh_graph.add_relationship("A", "KNOWS", "B", "Alice", "Bob", fact_ids=["f1", "f2"])
    assert "f1" in fresh_graph._fact_edge_index
    assert "f2" in fresh_graph._fact_edge_index


def test_remove_fact_reference_drops_correct_edge(fresh_graph):
    """remove_fact_reference must delete only the edge backed by the given fact."""
    fresh_graph.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    fresh_graph.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f2"])

    removed = fresh_graph.remove_fact_reference("f1")

    assert removed == 1
    edges = list(fresh_graph.G.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]["relation"] == "OWNS"


def test_remove_fact_reference_partial_clear(fresh_graph):
    """If an edge has two fact_ids, removing one should keep the edge alive."""
    fresh_graph.add_relationship("A", "OWNS", "B", "Alice", "Bakery", fact_ids=["f1", "f2"])

    removed = fresh_graph.remove_fact_reference("f1")

    assert removed == 0  # edge still has f2, so it is NOT deleted
    edges = list(fresh_graph.G.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]["source_fact_ids"] == ["f2"]


def test_fact_spanning_two_edges_removes_only_empty_one(fresh_graph):
    """A fact shared across two edges removes only the edge that becomes empty."""
    fresh_graph.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    fresh_graph.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f1", "f2"])

    removed = fresh_graph.remove_fact_reference("f1")

    assert removed == 1  # WORKS_AT has only f1 → deleted; OWNS still has f2 → kept
    edges = list(fresh_graph.G.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]["relation"] == "OWNS"
    assert edges[0][2]["source_fact_ids"] == ["f2"]


def test_remove_fact_reference_unknown_fact_is_noop(fresh_graph):
    fresh_graph.add_relationship("A", "KNOWS", "B", "Alice", "Bob", fact_ids=["f1"])
    removed = fresh_graph.remove_fact_reference("unknown_fact")
    assert removed == 0
    assert len(list(fresh_graph.G.edges())) == 1


def test_remove_fact_reference_clears_index_entry(fresh_graph):
    fresh_graph.add_relationship("A", "KNOWS", "B", "Alice", "Bob", fact_ids=["f1"])
    fresh_graph.remove_fact_reference("f1")
    assert "f1" not in fresh_graph._fact_edge_index


# ---------------------------------------------------------------------------
# Persistence (JSON round-trip)
# ---------------------------------------------------------------------------

def test_json_roundtrip_preserves_multi_edges(graph_path):
    g = KnowledgeRelationshipGraph(graph_path)
    g.add_relationship("X", "HAS",   "Y", "Xray", "Yacht", fact_ids=["f1"])
    g.add_relationship("X", "NEEDS", "Y", "Xray", "Yacht", fact_ids=["f2"])

    g2 = KnowledgeRelationshipGraph(graph_path)

    edges = list(g2.G.edges(data=True))
    assert len(edges) == 2
    predicates = {d["relation"] for _, _, d in edges}
    assert predicates == {"HAS", "NEEDS"}


def test_json_roundtrip_rebuilds_fact_index(graph_path):
    """Index must be correctly rebuilt from disk on load."""
    g = KnowledgeRelationshipGraph(graph_path)
    g.add_relationship("X", "HAS", "Y", "Xray", "Yacht", fact_ids=["f1"])

    g2 = KnowledgeRelationshipGraph(graph_path)
    assert "f1" in g2._fact_edge_index
    s, t, _ = g2._fact_edge_index["f1"][0]
    assert s == "X" and t == "Y"


def test_legacy_digraph_json_upgrades_to_multidigraph(graph_path):
    """Loading a DiGraph-format JSON must transparently produce a MultiDiGraph."""
    legacy_G = nx.DiGraph()
    legacy_G.add_node("A", name="Alice")
    legacy_G.add_node("B", name="Bob")
    legacy_G.add_edge("A", "B", relation="KNOWS", source_fact_ids=["f1"])
    with open(graph_path, "w") as f:
        from networkx.readwrite import json_graph
        json.dump(json_graph.node_link_data(legacy_G), f)

    g = KnowledgeRelationshipGraph(graph_path)

    assert isinstance(g.G, nx.MultiDiGraph)
    edges = list(g.G.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]["relation"] == "KNOWS"
    assert "f1" in g._fact_edge_index


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_wipes_graph_and_index(fresh_graph):
    fresh_graph.add_relationship("A", "LIKES", "B", "Alice", "Bob", fact_ids=["f1", "f2"])
    assert fresh_graph._fact_edge_index

    fresh_graph.clear()

    assert list(fresh_graph.G.nodes()) == []
    assert fresh_graph._fact_edge_index == {}


def test_clear_persists_empty_graph(graph_path):
    g = KnowledgeRelationshipGraph(graph_path)
    g.add_relationship("A", "LIKES", "B", "Alice", "Bob", fact_ids=["f1"])
    g.clear()

    g2 = KnowledgeRelationshipGraph(graph_path)
    assert list(g2.G.nodes()) == []
    assert g2._fact_edge_index == {}


# ---------------------------------------------------------------------------
# remove_relationship
# ---------------------------------------------------------------------------

def test_remove_relationship_cleans_index(fresh_graph):
    fresh_graph.add_relationship("A", "KNOWS", "B", "Alice", "Bob", fact_ids=["f1"])
    fresh_graph.remove_relationship("A", "B")

    assert list(fresh_graph.G.edges()) == []
    assert "f1" not in fresh_graph._fact_edge_index


def test_remove_relationship_removes_all_predicates(fresh_graph):
    fresh_graph.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    fresh_graph.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f2"])

    fresh_graph.remove_relationship("A", "B")

    assert list(fresh_graph.G.edges()) == []
    assert "f1" not in fresh_graph._fact_edge_index
    assert "f2" not in fresh_graph._fact_edge_index


def test_remove_relationship_noop_on_missing_edge(fresh_graph):
    fresh_graph.remove_relationship("nonexistent_A", "nonexistent_B")
