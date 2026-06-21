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
    AtomicFact,
    ContextHint,
    EntityExtraction,
    Entity,
    MemoryProcessing,
    KnowledgeTriple,
    MergeDecision,
    SplitDecision,
    SupersessionDecision,
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

    def test_add_graph_flushed_once_for_multiple_triples(self, app_client, monkeypatch):
        """write_graph() must be called exactly once per /memory/add, not once per triple."""
        import memory_server
        write_calls: list[int] = []
        real_write = memory_server.knowledge_graph.write_graph
        monkeypatch.setattr(
            memory_server.knowledge_graph,
            "write_graph",
            lambda: (write_calls.append(1), real_write())[1],
        )
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["A fact."],
                triples=[
                    KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery"),
                    KnowledgeTriple(subject="Alice", predicate="IS", object="Person"),
                    KnowledgeTriple(subject="Bakery", predicate="IS", object="Business"),
                ],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery."})
        assert len(write_calls) == 1


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
        assert "pruned" in report
        assert "merged" in report
        assert "split" in report
        assert "superseded" in report
        assert "flagged" in report
        assert isinstance(report["flagged"], list)

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

    def test_consolidate_phase4_was_facts_marked_historical(self, app_client, monkeypatch):
        """When IS and WAS edges exist for the same entity pair, WAS source facts are marked historical."""
        import memory_server

        call_count = [0]

        def vary_triple(_text):
            call_count[0] += 1
            pred = "WAS" if call_count[0] == 1 else "IS"
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=f"Hailey {pred.lower()} a fencer.")],
                triples=[KnowledgeTriple(subject="Hailey", predicate=pred, object="Fencer")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_triple)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)

        resp = app_client.post("/memory/consolidate")
        assert resp.status_code == 200
        assert resp.json()["report"]["superseded"] >= 1

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT content, temporal_status FROM atomic_facts")
        rows = dict(cursor.fetchall())
        assert rows.get("Hailey was a fencer.") == "historical"
        assert rows.get("Hailey is a fencer.") == "current"

    def test_consolidate_phase4_contradiction_flagged(self, app_client, monkeypatch):
        """IS and IS_NOT edges for the same entity pair generate a contradiction flag."""
        import memory_server

        call_count = [0]

        def vary_triple(_text):
            call_count[0] += 1
            pred = "IS" if call_count[0] == 1 else "IS_NOT"
            text = "Alice is a teacher." if pred == "IS" else "Alice is not a teacher."
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=text)],
                triples=[KnowledgeTriple(subject="Alice", predicate=pred, object="Teacher")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_triple)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)

        resp = app_client.post("/memory/consolidate")
        assert resp.status_code == 200
        flagged = resp.json()["report"]["flagged"]
        assert len(flagged) >= 1
        entry = flagged[0]
        assert entry["type"] == "contradiction"
        assert entry["predicate_a"] == "IS"
        assert entry["predicate_b"] == "IS_NOT"
        assert entry["subject"] == "Alice"
        assert entry["object"] == "Teacher"

    def test_consolidate_phase4_contradiction_not_duplicated_across_passes(self, app_client, monkeypatch):
        """A contradiction pair is flagged exactly once even when CONSOLIDATION_PASSES > 1."""
        import memory_server

        call_count = [0]

        def vary_triple(_text):
            call_count[0] += 1
            pred = "IS" if call_count[0] == 1 else "IS_NOT"
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=f"Bob {'is' if pred == 'IS' else 'is not'} tall.")],
                triples=[KnowledgeTriple(subject="Bob", predicate=pred, object="Tall")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_triple)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)

        resp = app_client.post("/memory/consolidate")
        flagged = resp.json()["report"]["flagged"]
        assert len([f for f in flagged if f["subject"] == "Bob"]) == 1

    def test_consolidate_phase4_supersession_chunk_granularity(self, app_client, monkeypatch):
        """
        KNOWN LIMITATION (pinned): supersession marks all facts in a chunk as historical,
        not just the fact that backs the WAS edge. Fact-to-edge linkage is chunk-level
        because the Librarian returns atomic_facts and triples as independent lists.

        Here "Hailey owns a dog." incorrectly becomes historical because it shares a
        batch with the WAS-Fencer edge. Per-triple fact linking would fix this but
        requires a Librarian schema change.
        """
        import memory_server

        call_count = [0]

        def two_fact_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[
                        AtomicFact(text="Hailey was a fencer."),
                        AtomicFact(text="Hailey owns a dog."),
                    ],
                    triples=[
                        KnowledgeTriple(subject="Hailey", predicate="WAS", object="Fencer"),
                        KnowledgeTriple(subject="Hailey", predicate="OWNS", object="Dog"),
                    ],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Hailey is a fencer.")],
                triples=[KnowledgeTriple(subject="Hailey", predicate="IS", object="Fencer")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", two_fact_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)

        app_client.post("/memory/consolidate")

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT content, temporal_status FROM atomic_facts")
        rows = dict(cursor.fetchall())
        assert rows.get("Hailey is a fencer.") == "current"
        # Chunk-granularity limitation: "Hailey owns a dog." shares batch fact_ids
        # with the WAS-Fencer edge and is incorrectly marked historical.
        assert rows.get("Hailey owns a dog.") == "historical"


# ---------------------------------------------------------------------------
# Temporal status: parse-time tagging and /context filtering
# ---------------------------------------------------------------------------

class TestTemporalStatus:
    def test_add_writes_temporal_status(self, app_client, monkeypatch):
        """temporal_status from AtomicFact is persisted to the atomic_facts table."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=[
                    AtomicFact(text="Alice owns a bakery.", temporal_status="current"),
                    AtomicFact(text="Alice used to live in Paris.", temporal_status="historical",
                               valid_period="college"),
                ],
                triples=[],
            ),
        )
        app_client.post("/memory/add", json={"text": "some text"})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT content, temporal_status, valid_period FROM atomic_facts ORDER BY created_at")
        rows = {r[0]: (r[1], r[2]) for r in cursor.fetchall()}
        assert rows["Alice owns a bakery."] == ("current", None)
        assert rows["Alice used to live in Paris."] == ("historical", "college")

    def test_add_string_facts_default_to_current(self, app_client, monkeypatch):
        """Plain string atomic_facts (backward-compat) are stored with temporal_status='current'."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["A timeless fact."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "some text"})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT temporal_status FROM atomic_facts WHERE content = 'A timeless fact.'")
        row = cursor.fetchone()
        assert row is not None and row[0] == "current"

    def test_context_excludes_historical_facts(self, app_client, monkeypatch):
        """Historical facts are excluded from /memory/context results."""
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=[
                    AtomicFact(text="Alice owns a bakery.", temporal_status="current"),
                    AtomicFact(text="Alice used to live in Paris.", temporal_status="historical"),
                ],
                triples=[],
            ),
        )
        app_client.post("/memory/add", json={"text": "some text"})

        resp = app_client.post("/memory/context", json={"query": "Alice"})
        assert resp.status_code == 200
        result_texts = [r["text"] for r in resp.json()["results"]]
        assert "Alice owns a bakery." in result_texts
        assert "Alice used to live in Paris." not in result_texts

    def test_context_does_not_bump_hit_count_for_historical(self, app_client, monkeypatch):
        """hit_count must not be incremented for historical facts surfaced by vector search."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=[AtomicFact(text="Old news.", temporal_status="historical")],
                triples=[],
            ),
        )
        app_client.post("/memory/add", json={"text": "some text"})
        app_client.post("/memory/context", json={"query": "Old news"})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT hit_count FROM atomic_facts WHERE content = 'Old news.'")
        row = cursor.fetchone()
        assert row is not None and row[0] == 0


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


# ---------------------------------------------------------------------------
# _split_into_chunks  (pure function, no server needed)
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    """Unit tests for the sentence-boundary chunker (no app_client required)."""

    @staticmethod
    def fn():
        import memory_server
        return memory_server._split_into_chunks

    def test_empty_text_returns_empty_list(self):
        assert self.fn()("", 5) == []

    def test_whitespace_only_returns_empty_list(self):
        assert self.fn()("   ", 5) == []

    def test_single_sentence_returns_one_chunk(self):
        result = self.fn()("Alice owns a bakery.", 5)
        assert result == ["Alice owns a bakery."]

    def test_fewer_sentences_than_chunk_size_returns_one_chunk(self):
        result = self.fn()("S1. S2. S3.", 5)
        assert len(result) == 1
        assert "S1" in result[0] and "S3" in result[0]

    def test_exact_chunk_size_returns_one_chunk(self):
        sentences = " ".join(f"S{i}." for i in range(5))
        assert len(self.fn()(sentences, 5)) == 1

    def test_double_chunk_size_returns_two_chunks_with_no_overlap(self):
        sentences = " ".join(f"S{i}." for i in range(10))
        result = self.fn()(sentences, 5)
        assert len(result) == 2
        assert "S4" not in result[1]  # no sentence duplicated between chunks

    def test_partial_last_chunk_included(self):
        sentences = " ".join(f"S{i}." for i in range(7))
        result = self.fn()(sentences, 5)
        assert len(result) == 2
        assert "S5" in result[1] and "S6" in result[1]

    def test_exclamation_and_question_marks_split(self):
        text = "Alice is great! Is she? She really is."
        result = self.fn()(text, 5)
        assert len(result) == 1  # 3 sentences < chunk_size=5, all in one chunk
        assert "Alice is great" in result[0]


# ---------------------------------------------------------------------------
# POST /memory/learn
# ---------------------------------------------------------------------------

class TestLearnEndpoint:

    def test_learn_returns_expected_keys(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Alice is kind."], triples=[]),
        )
        resp = app_client.post("/memory/learn", json={"text": "Alice is kind. She is generous."})
        assert resp.status_code == 200
        body = resp.json()
        for key in ("status", "chunks_total", "chunks_succeeded", "facts_added", "triples_added", "errors"):
            assert key in body, f"Missing key: {key}"

    def test_learn_empty_text_returns_zero_chunks(self, app_client):
        resp = app_client.post("/memory/learn", json={"text": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["chunks_total"] == 0
        assert body["facts_added"] == 0
        assert body["errors"] == []

    def test_learn_single_sentence_produces_one_chunk(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Alice is kind."], triples=[]),
        )
        resp = app_client.post("/memory/learn", json={"text": "Alice is kind."})
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_total"] == 1
        assert body["chunks_succeeded"] == 1
        assert body["facts_added"] == 1

    def test_learn_aggregates_facts_across_chunks(self, app_client, monkeypatch):
        """10 sentences → 2 chunks of 5 → each chunk produces 1 fact → facts_added == 2."""
        call_count = {"n": 0}

        def stub(text):
            call_count["n"] += 1
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=f"Fact {call_count['n']}.")],
                triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Kind")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", stub)
        sentences = " ".join(f"Sentence {i} is here." for i in range(10))
        resp = app_client.post("/memory/learn", json={"text": sentences})
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_total"] == 2
        assert body["chunks_succeeded"] == 2
        assert body["facts_added"] == 2
        assert body["triples_added"] == 2
        assert body["status"] == "success"

    def test_learn_partial_failure_returns_partial_status(self, app_client, monkeypatch):
        """If a chunk's Librarian call fails, the endpoint reports partial success."""
        call_count = {"n": 0}

        def stub(text):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return None  # triggers HTTPException in add_memory
            return MemoryProcessing(atomic_facts=["Alice is kind."], triples=[])

        monkeypatch.setattr("memory_server.process_memory_chunk", stub)
        sentences = " ".join(f"Sentence {i} is here." for i in range(10))
        resp = app_client.post("/memory/learn", json={"text": sentences})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "partial"
        assert body["chunks_total"] == 2
        assert body["chunks_succeeded"] == 1
        assert len(body["errors"]) == 1
        assert body["errors"][0]["chunk_index"] == 1

    def test_learn_no_overlap_between_chunks(self, app_client, monkeypatch):
        """Sentences are not duplicated across chunks — each sentence is ingested exactly once."""
        ingested: list[str] = []

        def stub(text):
            ingested.append(text)
            return MemoryProcessing(atomic_facts=[AtomicFact(text=text[:30] + ".")], triples=[])

        monkeypatch.setattr("memory_server.process_memory_chunk", stub)
        sentences = " ".join(f"S{i}." for i in range(10))
        app_client.post("/memory/learn", json={"text": sentences})

        # 2 chunks; no sentence appears in more than one chunk
        all_text = " ".join(ingested)
        for i in range(10):
            assert all_text.count(f"S{i}") == 1, f"S{i} ingested more than once"

    def test_learn_applies_context_hint_prefix_to_later_chunks(self, app_client, monkeypatch):
        """Hint from chunk 0 is prepended to chunks 1+, not to chunk 0 itself."""
        received: list[str] = []

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: (received.append(text), MemoryProcessing(atomic_facts=["Fact."], triples=[]))[1],
        )
        monkeypatch.setattr(
            "memory_server.extract_context_hint",
            lambda text: ContextHint(subject="Alice", time_period="college years"),
        )

        sentences = " ".join(f"S{i}." for i in range(10))
        resp = app_client.post("/memory/learn", json={"text": sentences})
        assert resp.status_code == 200

        # chunk 0: no prefix
        assert not received[0].startswith("[CONTEXT:")
        # chunk 1+: prefix present
        for text in received[1:]:
            assert text.startswith("[CONTEXT: Alice, college years]")

    def test_learn_skips_hint_extraction_for_single_chunk(self, app_client, monkeypatch):
        """extract_context_hint must not be called when there is only one chunk."""
        hint_calls: list[str] = []

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Fact."], triples=[]),
        )
        monkeypatch.setattr(
            "memory_server.extract_context_hint",
            lambda text: hint_calls.append(text) or ContextHint(),
        )

        resp = app_client.post("/memory/learn", json={"text": "Just one sentence."})
        assert resp.status_code == 200
        assert hint_calls == []

    def test_learn_proceeds_without_prefix_when_hint_returns_none(self, app_client, monkeypatch):
        """If extract_context_hint returns None, chunks are processed without any prefix."""
        received: list[str] = []

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: (received.append(text), MemoryProcessing(atomic_facts=["Fact."], triples=[]))[1],
        )
        # conftest default already stubs to None, but be explicit
        monkeypatch.setattr("memory_server.extract_context_hint", lambda text: None)

        sentences = " ".join(f"S{i}." for i in range(10))
        app_client.post("/memory/learn", json={"text": sentences})
        for text in received:
            assert not text.startswith("[CONTEXT:")
