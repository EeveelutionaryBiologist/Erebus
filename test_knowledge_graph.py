"""
Isolated tests for KnowledgeRelationshipGraph.
Run with: python test_knowledge_graph.py
No server or GGUF models required.
"""
import json
import tempfile
import os
import networkx as nx
from networkx.readwrite import json_graph
from knowledge_graph import KnowledgeRelationshipGraph


def make_graph(path):
    return KnowledgeRelationshipGraph(path)


def test_two_predicates_same_pair(path):
    """Two distinct predicates between the same entity pair must both survive."""
    g = make_graph(path)
    g.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    g.add_relationship("A", "OWNS", "B", "Alice", "Bakery", fact_ids=["f2"])

    edges = list(g.G.edges(data=True))
    assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}"
    predicates = {d['relation'] for _, _, d in edges}
    assert predicates == {"WORKS_AT", "OWNS"}, f"Predicates: {predicates}"
    print("PASS  test_two_predicates_same_pair")


def test_same_predicate_appends_facts(path):
    """Same predicate re-added should append fact_ids, not create a second edge."""
    g = make_graph(path)
    g.add_relationship("A", "LIKES", "C", "Alice", "Coffee", fact_ids=["f1"])
    g.add_relationship("A", "LIKES", "C", "Alice", "Coffee", fact_ids=["f2"])

    edges = list(g.G.edges(data=True))
    assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"
    fids = edges[0][2]['source_fact_ids']
    assert set(fids) == {"f1", "f2"}, f"fact_ids: {fids}"
    print("PASS  test_same_predicate_appends_facts")


def test_remove_fact_reference_drops_correct_edge(path):
    """remove_fact_reference removes only the edge backed by the deleted fact."""
    g = make_graph(path)
    g.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    g.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f2"])

    removed = g.remove_fact_reference("f1")
    assert removed == 1, f"Expected 1 edge removed, got {removed}"
    edges = list(g.G.edges(data=True))
    assert len(edges) == 1, f"Expected 1 edge remaining, got {len(edges)}"
    assert edges[0][2]['relation'] == "OWNS", f"Wrong edge survived: {edges[0][2]['relation']}"
    print("PASS  test_remove_fact_reference_drops_correct_edge")


def test_fact_spanning_two_edges(path):
    """A fact_id referenced by two edges only removes the edge(s) that become empty."""
    g = make_graph(path)
    g.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    g.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f1", "f2"])

    # Removing f1: WORKS_AT has only f1 (→ deleted); OWNS still has f2 (→ kept).
    removed = g.remove_fact_reference("f1")
    assert removed == 1, f"Expected 1 edge removed, got {removed}"
    edges = list(g.G.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]['relation'] == "OWNS"
    assert edges[0][2]['source_fact_ids'] == ["f2"]
    print("PASS  test_fact_spanning_two_edges")


def test_json_roundtrip_preserves_multi_edges(path):
    """Write → read cycle must preserve multiple edges between the same pair."""
    g = make_graph(path)
    g.add_relationship("X", "HAS",   "Y", "Xray", "Yacht", fact_ids=["f1"])
    g.add_relationship("X", "NEEDS", "Y", "Xray", "Yacht", fact_ids=["f2"])
    g.write_graph()

    g2 = make_graph(path)
    edges = list(g2.G.edges(data=True))
    assert len(edges) == 2, f"Round-trip lost edges: {len(edges)}"
    predicates = {d['relation'] for _, _, d in edges}
    assert predicates == {"HAS", "NEEDS"}
    # Index rebuilt from file must also be correct.
    assert ("X", "Y") in [(s, t) for s, t, _ in g2._fact_edge_index.get("f1", [])]
    print("PASS  test_json_roundtrip_preserves_multi_edges")


def test_legacy_digraph_json_upgrade(path):
    """Loading a DiGraph-format JSON file must transparently upgrade to MultiDiGraph."""
    legacy_G = nx.DiGraph()
    legacy_G.add_node("A", name="Alice")
    legacy_G.add_node("B", name="Bob")
    legacy_G.add_edge("A", "B", relation="KNOWS", source_fact_ids=["f1"])
    with open(path, 'w') as f:
        json.dump(json_graph.node_link_data(legacy_G), f)

    g = make_graph(path)
    assert isinstance(g.G, nx.MultiDiGraph), "Graph was not upgraded to MultiDiGraph"
    edges = list(g.G.edges(data=True))
    assert len(edges) == 1
    assert edges[0][2]['relation'] == "KNOWS"
    # Index should be populated from the legacy data.
    assert "f1" in g._fact_edge_index
    print("PASS  test_legacy_digraph_json_upgrade")


def test_clear_resets_index(path):
    """clear() must wipe both the graph and the fact→edge index."""
    g = make_graph(path)
    g.add_relationship("A", "LIKES", "B", "Alice", "Bob", fact_ids=["f1", "f2"])
    assert g._fact_edge_index  # non-empty before clear
    g.clear()
    assert not g.G.nodes(), "Graph still has nodes after clear()"
    assert g._fact_edge_index == {}, f"Index not empty after clear(): {g._fact_edge_index}"
    print("PASS  test_clear_resets_index")


def test_retrieve_relationships_multi_predicate(path):
    """retrieve_relationships must return all predicate strings for a multi-edge pair."""
    g = make_graph(path)
    g.add_relationship("A", "WORKS_AT", "B", "Alice", "Bakery", fact_ids=["f1"])
    g.add_relationship("A", "OWNS",     "B", "Alice", "Bakery", fact_ids=["f2"])

    facts = g.retrieve_relationships("A", depth=1)
    assert len(facts) == 2, f"Expected 2 facts, got {facts}"
    assert any("WORKS_AT" in f for f in facts)
    assert any("OWNS" in f for f in facts)
    print("PASS  test_retrieve_relationships_multi_predicate")


if __name__ == "__main__":
    tests = [
        test_two_predicates_same_pair,
        test_same_predicate_appends_facts,
        test_remove_fact_reference_drops_correct_edge,
        test_fact_spanning_two_edges,
        test_json_roundtrip_preserves_multi_edges,
        test_legacy_digraph_json_upgrade,
        test_clear_resets_index,
        test_retrieve_relationships_multi_predicate,
    ]

    failed = 0
    for test_fn in tests:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            # Start each test with a clean file (no pre-existing graph).
            os.unlink(tmp_path)
            test_fn(tmp_path)
        except Exception as e:
            print(f"FAIL  {test_fn.__name__}: {e}")
            failed += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    print()
    total = len(tests)
    print(f"{total - failed}/{total} tests passed.")
    if failed:
        raise SystemExit(1)
