"""
Integration tests for memory_server FastAPI endpoints.

All tests use the `app_client` fixture from conftest, which provides:
  - Isolated SQLite, ChromaDB, and KnowledgeGraph in a tmp directory
  - get_embedding stubbed to return [0.1] * 768
  - All Librarian functions available for per-test patching via monkeypatch

Librarian stubs
---------------
Tests that exercise /memory/add or /memory/consolidate need to control what the
Librarian returns. Patch at the memory_server level, e.g.:

    from librarian import MemoryProcessing, KnowledgeTriple
    monkeypatch.setattr(
        "memory_server.process_memory_chunk",
        lambda text: MemoryProcessing(
            atomic_facts=["Alice owns a bakery."],
            triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
        ),
    )
"""

import pytest
from librarian import (
    EntityExtraction,
    Entity,
    MemoryProcessing,
    KnowledgeTriple,
    MergeDecision,
    SplitDecision,
)


# ---------------------------------------------------------------------------
# POST /memory/add
# ---------------------------------------------------------------------------

class TestAddMemory:
    def test_add_returns_success(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
            ),
        )
        resp = app_client.post("/memory/add", json={"text": "Alice owns a bakery."})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "1 standalone facts" in body["message"]

    def test_add_persists_raw_chunk(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["A fact."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "raw input"})

        resp = app_client.get("/memory/all")
        raws = [r for r in resp.json()["results"] if r["record_type"] == "raw"]
        assert any("raw input" in r["text"] for r in raws)

    def test_add_persists_atomic_fact(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["A stored fact."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "some input"})

        resp = app_client.get("/memory/all")
        facts = [r for r in resp.json()["results"] if r["record_type"] == "fact"]
        assert any("A stored fact." in r["text"] for r in facts)

    def test_add_creates_entity_and_graph_edge(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery."})

        resp = app_client.get("/memory/all")
        entities = [r for r in resp.json()["results"] if r["record_type"] == "entity"]
        names = {e["text"] for e in entities}
        assert "Alice" in names
        assert "Bakery" in names

    def test_add_librarian_failure_returns_500(self, app_client, monkeypatch):
        monkeypatch.setattr("memory_server.process_memory_chunk", lambda text: None)
        resp = app_client.post("/memory/add", json={"text": "anything"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /memory/search
# ---------------------------------------------------------------------------

class TestSearchMemory:
    def _seed(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Alice owns a bakery."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery."})

    def test_search_returns_results_key(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.extract_entities_from_text",
            lambda text: EntityExtraction(entities=[]),
        )
        self._seed(app_client, monkeypatch)
        resp = app_client.post("/memory/search", json={"query": "bakery", "top_k": 3})
        assert resp.status_code == 200
        assert "results" in resp.json()
        assert "relational_context" in resp.json()

    def test_search_increments_hit_count(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.extract_entities_from_text",
            lambda text: EntityExtraction(entities=[]),
        )
        self._seed(app_client, monkeypatch)
        app_client.post("/memory/search", json={"query": "Alice bakery", "top_k": 1})
        app_client.post("/memory/search", json={"query": "Alice bakery", "top_k": 1})

        resp = app_client.get("/memory/all")
        facts = [r for r in resp.json()["results"] if r["record_type"] == "fact"]
        assert any(r["hit_count"] >= 1 for r in facts)

    def test_search_graph_lookup_uses_entity_names(self, app_client, monkeypatch):
        # Seed a triple so there's something in the graph.
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery."})

        monkeypatch.setattr(
            "memory_server.extract_entities_from_text",
            lambda text: EntityExtraction(entities=[Entity(name="Alice")]),
        )
        resp = app_client.post("/memory/search", json={"query": "What does Alice own?", "top_k": 3})
        assert "Alice" in resp.json()["relational_context"]


# ---------------------------------------------------------------------------
# GET /memory/all
# ---------------------------------------------------------------------------

class TestGetAllMemories:
    def test_all_returns_three_record_types(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["A fact."],
                triples=[KnowledgeTriple(subject="X", predicate="IS", object="Y")],
            ),
        )
        app_client.post("/memory/add", json={"text": "some text"})

        resp = app_client.get("/memory/all")
        types = {r["record_type"] for r in resp.json()["results"]}
        assert types == {"raw", "fact", "entity"}

    def test_all_empty_on_fresh_db(self, app_client):
        resp = app_client.get("/memory/all")
        assert resp.status_code == 200
        assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# DELETE /memory/clear
# ---------------------------------------------------------------------------

class TestClearMemories:
    def test_clear_empties_all_tables(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["A fact."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "some text"})

        resp = app_client.delete("/memory/clear")
        assert resp.status_code == 200

        resp = app_client.get("/memory/all")
        assert resp.json()["results"] == []

    def test_clear_empties_knowledge_graph(self, app_client, monkeypatch):
        import memory_server
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["A fact."],
                triples=[KnowledgeTriple(subject="A", predicate="IS", object="B")],
            ),
        )
        app_client.post("/memory/add", json={"text": "A is B."})
        app_client.delete("/memory/clear")

        assert list(memory_server.knowledge_graph.G.nodes()) == []
        assert memory_server.knowledge_graph._fact_edge_index == {}


# ---------------------------------------------------------------------------
# POST /memory/consolidate
# ---------------------------------------------------------------------------

class TestConsolidateMemories:
    def test_consolidate_returns_report(self, app_client, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)

        resp = app_client.post("/memory/consolidate")
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert "pruned" in report and "merged" in report and "split" in report

    def test_consolidate_exact_dedup_removes_lower_hit_copy(self, app_client, monkeypatch):
        """Two identical facts added separately must collapse to one after consolidation."""
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Duplicate fact."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)

        app_client.post("/memory/consolidate")

        resp = app_client.get("/memory/all")
        facts = [r for r in resp.json()["results"] if r["record_type"] == "fact"]
        duplicate_facts = [f for f in facts if f["text"] == "Duplicate fact."]
        assert len(duplicate_facts) == 1
