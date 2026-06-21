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

    def _seed(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["A fact."],
                triples=[KnowledgeTriple(subject="X", predicate="IS", object="Y")],
            ),
        )
        app_client.post("/memory/add", json={"text": "some text"})

    def test_all_type_raw_returns_only_raw(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.get("/memory/all?type=raw")
        assert resp.status_code == 200
        types = {r["record_type"] for r in resp.json()["results"]}
        assert types == {"raw"}

    def test_all_type_fact_returns_only_facts(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.get("/memory/all?type=fact")
        assert resp.status_code == 200
        types = {r["record_type"] for r in resp.json()["results"]}
        assert types == {"fact"}

    def test_all_type_entity_returns_only_entities(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.get("/memory/all?type=entity")
        assert resp.status_code == 200
        types = {r["record_type"] for r in resp.json()["results"]}
        assert types == {"entity"}

    def test_all_invalid_type_returns_422(self, app_client):
        resp = app_client.get("/memory/all?type=garbage")
        assert resp.status_code == 422


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


# ---------------------------------------------------------------------------
# Entity / predicate normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    # -- Pure unit tests (no server, no fixtures needed) --------------------

    def test_entity_name_lowercased_input(self):
        from memory_server import normalize_entity_name
        assert normalize_entity_name("hailey") == "Hailey"

    def test_entity_name_allcaps_input(self):
        from memory_server import normalize_entity_name
        assert normalize_entity_name("ALICE SMITH") == "Alice Smith"

    def test_entity_name_already_canonical(self):
        from memory_server import normalize_entity_name
        assert normalize_entity_name("Alice") == "Alice"

    def test_entity_name_strips_whitespace(self):
        from memory_server import normalize_entity_name
        assert normalize_entity_name("  Bob  ") == "Bob"

    def test_predicate_uppercased(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("has") == "HAS"

    def test_predicate_spaces_become_underscores(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("works at") == "WORKS_AT"

    def test_predicate_synonym_has_a(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("has a") == "HAS"

    def test_predicate_synonym_is_a(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("is a") == "IS"

    def test_predicate_synonym_works_for(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("works for") == "WORKS_AT"

    def test_predicate_already_canonical(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("OWNS") == "OWNS"

    def test_predicate_idempotent(self):
        from memory_server import normalize_predicate
        assert normalize_predicate("WORKS_AT") == "WORKS_AT"

    # -- Integration tests (require app_client) -----------------------------

    def test_case_variant_entities_collapse_to_one_row(self, app_client, monkeypatch):
        """'hailey' and 'HAILEY' from the LLM should land in a single entity row."""
        call_count = [0]
        def variant_chunk(text):
            call_count[0] += 1
            name = "hailey" if call_count[0] == 1 else "HAILEY"
            return MemoryProcessing(
                atomic_facts=[f"{name} has a cat."],
                triples=[KnowledgeTriple(subject=name, predicate="HAS", object="Cat")],
            )
        monkeypatch.setattr("memory_server.process_memory_chunk", variant_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        resp = app_client.get("/memory/all")
        entities = [r for r in resp.json()["results"] if r["record_type"] == "entity"]
        hailey_rows = [e for e in entities if e["text"].lower() == "hailey"]
        assert len(hailey_rows) == 1

    def test_synonym_predicates_collapse_to_one_edge(self, app_client, monkeypatch):
        """'has a' and 'has' must map to one 'HAS' edge, not two separate edges."""
        import memory_server
        call_count = [0]
        def variant_chunk(text):
            call_count[0] += 1
            pred = "has a" if call_count[0] == 1 else "has"
            return MemoryProcessing(
                atomic_facts=["Alice has a cat."],
                triples=[KnowledgeTriple(subject="Alice", predicate=pred, object="Cat")],
            )
        monkeypatch.setattr("memory_server.process_memory_chunk", variant_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        kg = memory_server.knowledge_graph
        all_edges = list(kg.G.edges(data=True))
        has_edges = [e for e in all_edges if e[2].get("relation") == "HAS"]
        assert len(has_edges) == 1

    def test_normalized_entity_name_stored_in_sqlite(self, app_client, monkeypatch):
        """Canonical name stored in SQLite must be title-cased regardless of LLM output."""
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["mochi is a cat."],
                triples=[KnowledgeTriple(subject="mochi", predicate="IS", object="cat")],
            ),
        )
        app_client.post("/memory/add", json={"text": "mochi is a cat."})

        resp = app_client.get("/memory/all")
        entities = [r for r in resp.json()["results"] if r["record_type"] == "entity"]
        names = {e["text"] for e in entities}
        assert "Mochi" in names
        assert "Cat" in names


# ---------------------------------------------------------------------------
# POST /memory/context
# ---------------------------------------------------------------------------

class TestContextMemory:
    # -- Unit tests for the regex tokenizer (no server needed) --------------

    def test_extract_candidates_single_words(self):
        from memory_server import _extract_entity_candidates
        result = _extract_entity_candidates("Alice owns a bakery")
        assert "Alice" in result
        assert "bakery" in result

    def test_extract_candidates_bigrams(self):
        from memory_server import _extract_entity_candidates
        result = _extract_entity_candidates("Alice Smith owns a bakery")
        assert "Alice Smith" in result

    def test_extract_candidates_deduplicates(self):
        from memory_server import _extract_entity_candidates
        result = _extract_entity_candidates("alice alice")
        assert result.count("alice") == 1

    def test_extract_candidates_empty_string(self):
        from memory_server import _extract_entity_candidates
        assert _extract_entity_candidates("") == []

    # -- Integration tests --------------------------------------------------

    def _seed(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery."})

    def test_context_returns_expected_keys(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.post("/memory/context", json={"query": "Alice"})
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert "relational_context" in body

    def test_context_no_librarian_called(self, app_client, monkeypatch):
        """The /context endpoint must never touch the Librarian."""
        def should_not_be_called(*args, **kwargs):
            raise AssertionError("Librarian must not be called from /memory/context")

        for fn in ["process_memory_chunk", "extract_entities_from_text",
                   "librarian_should_merge", "librarian_split_compound"]:
            monkeypatch.setattr(f"memory_server.{fn}", should_not_be_called)

        resp = app_client.post("/memory/context", json={"query": "anything"})
        assert resp.status_code == 200

    def test_context_returns_vector_results(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.post("/memory/context", json={"query": "bakery ownership"})
        assert resp.status_code == 200
        # Stub embeddings are all [0.1]*768 so cosine similarity is 1.0 for every pair;
        # the seeded fact must appear in the results.
        assert len(resp.json()["results"]) >= 1

    def test_context_graph_lookup_finds_known_entity(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.post("/memory/context", json={"query": "What does Alice own?"})
        assert "Alice" in resp.json()["relational_context"]

    def test_context_graph_lookup_case_insensitive(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        resp = app_client.post("/memory/context", json={"query": "what does alice own?"})
        assert "Alice" in resp.json()["relational_context"]

    def test_context_increments_fact_hit_count(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        app_client.post("/memory/context", json={"query": "bakery"})

        resp = app_client.get("/memory/all")
        facts = [r for r in resp.json()["results"] if r["record_type"] == "fact"]
        assert any(r["hit_count"] >= 1 for r in facts)

    def test_context_increments_entity_hit_count(self, app_client, monkeypatch):
        self._seed(app_client, monkeypatch)
        app_client.post("/memory/context", json={"query": "Alice"})

        resp = app_client.get("/memory/all")
        entities = [r for r in resp.json()["results"] if r["record_type"] == "entity"]
        alice = next((e for e in entities if e["text"] == "Alice"), None)
        assert alice is not None and alice["hit_count"] >= 1

    def test_context_empty_db_returns_empty(self, app_client):
        resp = app_client.post("/memory/context", json={"query": "anything"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == []
        assert body["relational_context"] == ""
