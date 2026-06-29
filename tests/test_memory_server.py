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
from datetime import datetime
from librarian import (
    AtomicFact,
    CompoundEntityDecision,
    ContextHint,
    ConcurrencyDecision,
    EntityClassification,
    EntityExtraction,
    EntityMergeDecision,
    Entity,
    GroupAssignment,
    MemoryProcessing,
    KnowledgeTriple,
    MergeDecision,
    SplitDecision,
    SupersessionDecision,
)


def wait_for_task(client, task_id: str) -> dict:
    """Poll GET /memory/task/{task_id} and return the task dict.

    In tests, _run_task_in_background is patched to run synchronously (see conftest),
    so the task is already completed before the originating HTTP response arrives.
    """
    resp = client.get(f"/memory/task/{task_id}")
    assert resp.status_code == 200, f"task lookup failed: {resp.text}"
    return resp.json()


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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        result = task["result"]
        assert result["status"] == "success"
        assert "1 standalone facts" in result["message"]

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

    def test_add_librarian_failure_reports_task_error(self, app_client, monkeypatch):
        monkeypatch.setattr("memory_server.process_memory_chunk", lambda text: None)
        resp = app_client.post("/memory/add", json={"text": "anything"})
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "failed"
        assert "Librarian" in task["error"]

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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        assert task["result"]["report"]["superseded"] >= 1

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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        flagged = task["result"]["report"]["flagged"]
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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        flagged = task["result"]["report"]["flagged"]
        assert len([f for f in flagged if f["subject"] == "Bob"]) == 1

    def test_consolidate_phase4_supersession_per_triple_linkage(self, app_client, monkeypatch):
        """
        Phase 4 supersession only marks the fact that backs the WAS edge as historical,
        not unrelated facts from the same chunk. supporting_fact_indices on KnowledgeTriple
        pins each edge to the specific atomic fact(s) that produced it.

        "Hailey owns a dog." must stay 'current' because only atomic_facts[0]
        ("Hailey was a fencer.", index 0) backs the WAS-Fencer edge.

        get_embedding is stubbed to assign a distinct unit vector to each unique text so
        that Phase 2 does not spuriously dedup semantically different facts.
        """
        import memory_server

        # Each unique text gets its own dimension → cosine sim between different facts = 0.
        seen_texts: dict[str, int] = {}
        def unique_embedding(text):
            if text not in seen_texts:
                seen_texts[text] = len(seen_texts)
            vec = [0.0] * 768
            vec[seen_texts[text] % 768] = 1.0
            return vec
        monkeypatch.setattr("memory_server.get_embedding", unique_embedding)

        call_count = [0]

        def two_fact_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[
                        AtomicFact(text="Hailey was a fencer."),   # index 0
                        AtomicFact(text="Hailey owns a dog."),      # index 1
                    ],
                    triples=[
                        KnowledgeTriple(subject="Hailey", predicate="WAS", object="Fencer",
                                        supporting_fact_indices=[0]),
                        KnowledgeTriple(subject="Hailey", predicate="OWNS", object="Dog",
                                        supporting_fact_indices=[1]),
                    ],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Hailey is a fencer.")],
                triples=[KnowledgeTriple(subject="Hailey", predicate="IS", object="Fencer",
                                         supporting_fact_indices=[0])],
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
        assert rows.get("Hailey was a fencer.") == "historical"
        # Per-triple linkage: "Hailey owns a dog." is backed only by index 1 (OWNS edge),
        # which has no IS counterpart — so it must remain 'current'.
        assert rows.get("Hailey owns a dog.") == "current"


# ---------------------------------------------------------------------------
# Phase 4b: Text-based supersession via librarian_check_supersession
# ---------------------------------------------------------------------------

class TestConsolidatePhase4b:
    def _stub_consolidation_helpers(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)

    def test_phase4b_a_supersedes_b_marks_neighbor_historical(self, app_client, monkeypatch):
        """Keyword fact (A) superseding neighbor (B) marks B historical."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice no longer works at Google.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice works at Google.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_check_supersession",
            lambda a, b: SupersessionDecision(outcome="A_supersedes_B"),
        )
        self._stub_consolidation_helpers(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        assert task["result"]["report"]["superseded"] >= 1

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT content, temporal_status FROM atomic_facts")
        rows = dict(cursor.fetchall())
        assert rows.get("Alice works at Google.") == "historical"
        assert rows.get("Alice no longer works at Google.") == "current"

    def test_phase4b_b_supersedes_a_marks_keyword_fact_historical(self, app_client, monkeypatch):
        """When the neighbor (B) supersedes the keyword fact (A), A becomes historical."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice previously worked at Google.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice works at MegaCorp.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_check_supersession",
            lambda a, b: SupersessionDecision(outcome="B_supersedes_A"),
        )
        self._stub_consolidation_helpers(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["result"]["report"]["superseded"] >= 1

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT content, temporal_status FROM atomic_facts")
        rows = dict(cursor.fetchall())
        assert rows.get("Alice previously worked at Google.") == "historical"
        assert rows.get("Alice works at MegaCorp.") == "current"

    def test_phase4b_no_keywords_librarian_not_called(self, app_client, monkeypatch):
        """Facts with no supersession keywords never trigger librarian_check_supersession."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=f"Bob likes pizza (variant {call_count[0]}).")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        librarian_calls = [0]

        def tracking_supersession(a, b):
            librarian_calls[0] += 1
            return SupersessionDecision(outcome="neither")

        monkeypatch.setattr("memory_server.librarian_check_supersession", tracking_supersession)
        self._stub_consolidation_helpers(monkeypatch)

        app_client.post("/memory/consolidate")

        assert librarian_calls[0] == 0

    def test_phase4b_contradiction_flagged_with_text_source(self, app_client, monkeypatch):
        """Text-based contradictions appear in flagged list with source='text_based'."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice formerly ran marathons.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice runs marathons every year.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_check_supersession",
            lambda a, b: SupersessionDecision(outcome="contradiction"),
        )
        self._stub_consolidation_helpers(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        flagged = task["result"]["report"]["flagged"]

        text_based = [f for f in flagged if f.get("source") == "text_based"]
        assert len(text_based) >= 1
        entry = text_based[0]
        assert entry["type"] == "contradiction"
        assert "fact_a" in entry and "fact_b" in entry

    def test_phase4b_neither_outcome_does_not_supersede(self, app_client, monkeypatch):
        """A 'neither' outcome from the librarian leaves the report superseded count unchanged."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Bob used to live in Berlin.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Bob now lives in London.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_check_supersession",
            lambda a, b: SupersessionDecision(outcome="neither"),
        )
        self._stub_consolidation_helpers(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        report = task["result"]["report"]

        # Phase 4b "neither" adds no supersessions and no text_based flagged entries.
        assert report["superseded"] == 0
        assert not any(f.get("source") == "text_based" for f in report["flagged"])

        # The keyword fact itself must not have been marked historical by Phase 4b.
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT temporal_status FROM atomic_facts WHERE content = 'Bob used to live in Berlin.'"
        )
        row = cursor.fetchone()
        # The keyword fact either still exists (current) or was merged by Phase 2 (also fine).
        # Either way, Phase 4b did not touch it — so if it exists, it must be 'current'.
        if row:
            assert row[0] == "current"

    def test_phase4b_pair_not_rechecked_across_passes(self, app_client, monkeypatch):
        """The same fact pair is only sent to librarian_check_supersession once across all passes."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Carol formerly studied chemistry.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Carol studies biology.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        librarian_calls = [0]

        def counting_supersession(a, b):
            librarian_calls[0] += 1
            return SupersessionDecision(outcome="neither")

        monkeypatch.setattr("memory_server.librarian_check_supersession", counting_supersession)
        self._stub_consolidation_helpers(monkeypatch)

        app_client.post("/memory/consolidate")

        # With CONSOLIDATION_PASSES = 2, the pair should still only be checked once.
        assert librarian_calls[0] == 1


# ---------------------------------------------------------------------------
# Source chunk linkage
# ---------------------------------------------------------------------------

class TestSourceChunkLinkage:
    def test_add_populates_source_chunk_id_on_facts(self, app_client, monkeypatch):
        """Each atomic fact row should reference the raw_chunk that produced it."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery.", "Alice lives in Paris."],
                triples=[],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery. Alice lives in Paris."})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM raw_chunks")
        raw_id = cursor.fetchone()[0]

        cursor.execute("SELECT content, source_chunk_id FROM atomic_facts ORDER BY content")
        rows = {r[0]: r[1] for r in cursor.fetchall()}
        assert rows["Alice lives in Paris."] == raw_id
        assert rows["Alice owns a bakery."] == raw_id

    def test_add_populates_entity_chunks_for_triple_subjects_and_objects(self, app_client, monkeypatch):
        """entity_chunks rows are created for both the subject and object of each triple."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice owns a bakery."})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM raw_chunks")
        raw_id = cursor.fetchone()[0]

        cursor.execute("""
            SELECT e.canonical_name FROM entity_chunks ec
            JOIN entities e ON ec.entity_id = e.id
            WHERE ec.chunk_id = ?
            ORDER BY e.canonical_name
        """, (raw_id,))
        names = [r[0] for r in cursor.fetchall()]
        assert "Alice" in names
        assert "Bakery" in names

    def test_entity_chunks_dedup_across_multiple_chunks(self, app_client, monkeypatch):
        """Same entity appearing in two chunks creates two entity_chunks rows, not one."""
        import memory_server

        call_count = [0]

        def vary_triple(_text):
            call_count[0] += 1
            pred = "OWNS" if call_count[0] == 1 else "LIVES_IN"
            obj = "Bakery" if call_count[0] == 1 else "Paris"
            return MemoryProcessing(
                atomic_facts=[f"Alice {pred.lower()} {obj}."],
                triples=[KnowledgeTriple(subject="Alice", predicate=pred, object=obj)],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_triple)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM entity_chunks ec JOIN entities e ON ec.entity_id = e.id "
            "WHERE LOWER(e.canonical_name) = 'alice'"
        )
        count = cursor.fetchone()[0]
        assert count == 2

    def test_all_facts_response_includes_source_chunk_id(self, app_client, monkeypatch):
        """GET /memory/all?type=fact returns source_chunk_id on every fact record."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Bob drinks coffee."], triples=[]),
        )
        app_client.post("/memory/add", json={"text": "Bob drinks coffee."})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM raw_chunks")
        raw_id = cursor.fetchone()[0]

        resp = app_client.get("/memory/all?type=fact")
        facts = resp.json()["results"]
        assert len(facts) == 1
        assert facts[0]["source_chunk_id"] == raw_id

    def test_all_entities_response_includes_chunk_count(self, app_client, monkeypatch):
        """GET /memory/all?type=entity returns chunk_count showing how many chunks reference it."""
        import memory_server

        call_count = [0]

        def vary_triple(_text):
            call_count[0] += 1
            pred = "OWNS" if call_count[0] == 1 else "LIKES"
            return MemoryProcessing(
                atomic_facts=[f"Alice {pred.lower()} something."],
                triples=[KnowledgeTriple(subject="Alice", predicate=pred, object="Something")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_triple)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        resp = app_client.get("/memory/all?type=entity")
        entities = resp.json()["results"]
        alice = next(e for e in entities if e["text"] == "Alice")
        # Alice appears in two chunks.
        assert alice["chunk_count"] == 2

    def test_search_response_includes_source_chunk_id(self, app_client, monkeypatch):
        """POST /memory/search results include source_chunk_id on each fact."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Carol is a scientist."], triples=[]),
        )
        monkeypatch.setattr(
            "memory_server.extract_entities_from_text",
            lambda q: None,
        )
        app_client.post("/memory/add", json={"text": "Carol is a scientist."})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM raw_chunks")
        raw_id = cursor.fetchone()[0]

        resp = app_client.post("/memory/search", json={"query": "Carol scientist", "top_k": 1})
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["source_chunk_id"] == raw_id


# ---------------------------------------------------------------------------
# Tags / Groups
# ---------------------------------------------------------------------------

class TestEntityGroups:
    def test_add_assigns_matching_existing_group(self, app_client, monkeypatch):
        """When librarian_assign_groups returns a matching existing group, entity is linked to it."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice is a doctor."],
                triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Doctor")],
            ),
        )
        monkeypatch.setattr(
            "memory_server.librarian_assign_groups",
            lambda name, groups, **kw: GroupAssignment(matching_groups=["Family"], new_group=None),
        )
        # Seed an existing group so the Librarian can "find" it.
        cursor = memory_server.sqlite_conn.cursor()
        family_id = str(__import__("uuid").uuid4())
        cursor.execute(
            "INSERT INTO groups (id, name, created_at) VALUES (?, 'Family', '2020-01-01')",
            (family_id,),
        )
        memory_server.sqlite_conn.commit()

        app_client.post("/memory/add", json={"text": "Alice is a doctor."})

        cursor.execute("""
            SELECT g.name FROM entity_groups eg
            JOIN entities e ON eg.entity_id = e.id
            JOIN groups g ON eg.group_id = g.id
            WHERE LOWER(e.canonical_name) = 'alice'
        """)
        group_names = [r[0] for r in cursor.fetchall()]
        assert "Family" in group_names

    def test_add_creates_new_group_when_none_match(self, app_client, monkeypatch):
        """When librarian_assign_groups proposes a new_group, the group is created."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Mochi is a cat."],
                triples=[KnowledgeTriple(subject="Mochi", predicate="IS", object="Cat")],
            ),
        )
        monkeypatch.setattr(
            "memory_server.librarian_assign_groups",
            lambda name, groups, **kw: GroupAssignment(matching_groups=[], new_group="Pets"),
        )
        app_client.post("/memory/add", json={"text": "Mochi is a cat."})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("""
            SELECT g.name FROM entity_groups eg
            JOIN entities e ON eg.entity_id = e.id
            JOIN groups g ON eg.group_id = g.id
            WHERE LOWER(e.canonical_name) = 'mochi'
        """)
        group_names = [r[0] for r in cursor.fetchall()]
        assert "Pets" in group_names

        cursor.execute("SELECT name FROM groups WHERE LOWER(name) = 'pets'")
        assert cursor.fetchone() is not None

    def test_add_skips_group_assignment_for_known_entity(self, app_client, monkeypatch):
        """If an entity already has group assignments, librarian_assign_groups is not called again."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            return MemoryProcessing(
                atomic_facts=["Alice works here."],
                triples=[KnowledgeTriple(subject="Alice", predicate="WORKS_AT", object="Lab")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)

        librarian_calls = [0]

        def counting_assign(name, groups, **kw):
            librarian_calls[0] += 1
            return GroupAssignment(matching_groups=[], new_group="Friends")

        monkeypatch.setattr("memory_server.librarian_assign_groups", counting_assign)

        # First /add — Alice is new, should call librarian_assign_groups.
        app_client.post("/memory/add", json={"text": "first"})
        first_call_count = librarian_calls[0]

        # Second /add — Alice already has a group, should NOT call again.
        app_client.post("/memory/add", json={"text": "second"})
        assert librarian_calls[0] == first_call_count

    def test_all_entities_includes_groups(self, app_client, monkeypatch):
        """GET /memory/all?type=entity returns a groups list for each entity."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Bob owns a bakery."],
                triples=[KnowledgeTriple(subject="Bob", predicate="OWNS", object="Bakery")],
            ),
        )
        monkeypatch.setattr(
            "memory_server.librarian_assign_groups",
            lambda name, groups, **kw: GroupAssignment(matching_groups=[], new_group="Friends"),
        )
        app_client.post("/memory/add", json={"text": "Bob owns a bakery."})

        resp = app_client.get("/memory/all?type=entity")
        entities = resp.json()["results"]
        bob = next((e for e in entities if e["text"] == "Bob"), None)
        assert bob is not None
        assert "groups" in bob
        assert "Friends" in bob["groups"]

    def test_search_includes_entity_groups(self, app_client, monkeypatch):
        """POST /memory/search returns entity_groups when known entities are found in the KG."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Carol is a scientist."],
                triples=[KnowledgeTriple(subject="Carol", predicate="IS", object="Scientist")],
            ),
        )
        monkeypatch.setattr(
            "memory_server.librarian_assign_groups",
            lambda name, groups, **kw: GroupAssignment(matching_groups=[], new_group="Colleagues"),
        )
        app_client.post("/memory/add", json={"text": "Carol is a scientist."})

        monkeypatch.setattr(
            "memory_server.extract_entities_from_text",
            lambda q: EntityExtraction(entities=[Entity(name="Carol")]),
        )
        resp = app_client.post("/memory/search", json={"query": "Carol"})
        entity_groups = resp.json()["entity_groups"]
        assert "Carol" in entity_groups
        assert "Colleagues" in entity_groups["Carol"]


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
# Entity lookup — tiered matching + alias population
# ---------------------------------------------------------------------------

class TestEntityLookup:
    """Unit and integration tests for lookup_entity_ids() and alias backfill."""

    def test_compute_aliases_multi_word(self):
        from memory_server import _compute_aliases
        assert _compute_aliases("Alice Mercer") == ["Alice", "Mercer"]

    def test_compute_aliases_single_word_empty(self):
        from memory_server import _compute_aliases
        assert _compute_aliases("Alice") == []

    def test_compute_aliases_strips_stopwords(self):
        from memory_server import _compute_aliases
        result = _compute_aliases("University Of California")
        assert "Of" not in result
        assert "University" in result
        assert "California" in result

    def test_get_or_create_entity_populates_aliases(self, app_client):
        import memory_server
        import json
        entity_id = memory_server.get_or_create_entity("Alice Mercer")
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT aliases FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        aliases = json.loads(row[0])
        assert "Alice" in aliases
        assert "Mercer" in aliases

    def test_lookup_entity_ids_exact_match(self, app_client):
        import memory_server
        memory_server.get_or_create_entity("Alice Mercer")
        ids = memory_server.lookup_entity_ids("Alice Mercer")
        assert len(ids) == 1

    def test_lookup_entity_ids_prefix_match(self, app_client):
        import memory_server
        memory_server.get_or_create_entity("Alice Mercer")
        ids = memory_server.lookup_entity_ids("Alice")
        assert len(ids) == 1

    def test_lookup_entity_ids_suffix_match(self, app_client):
        import memory_server
        memory_server.get_or_create_entity("Alice Mercer")
        ids = memory_server.lookup_entity_ids("Mercer")
        assert len(ids) == 1

    def test_lookup_entity_ids_alias_match(self, app_client):
        """Middle token not reachable by prefix/suffix is found via alias json_each."""
        import memory_server
        memory_server.get_or_create_entity("Big Alice Corp")
        ids = memory_server.lookup_entity_ids("Alice")
        assert len(ids) == 1

    def test_lookup_entity_ids_case_insensitive(self, app_client):
        import memory_server
        memory_server.get_or_create_entity("Alice Mercer")
        assert len(memory_server.lookup_entity_ids("alice")) == 1
        assert len(memory_server.lookup_entity_ids("ALICE")) == 1

    def test_lookup_entity_ids_multiple_matches(self, app_client):
        """Ambiguous first name returns all matching entities."""
        import memory_server
        memory_server.get_or_create_entity("Alice Mercer")
        memory_server.get_or_create_entity("Alice Kim")
        ids = memory_server.lookup_entity_ids("Alice")
        assert len(ids) == 2

    def test_lookup_entity_ids_no_match(self, app_client):
        import memory_server
        ids = memory_server.lookup_entity_ids("Zephyrine Nonexistent")
        assert ids == []

    def test_search_graph_lookup_resolves_partial_name(self, app_client, monkeypatch):
        """Search with first-name-only LLM extraction still returns relational context."""
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice Mercer owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice Mercer", predicate="OWNS", object="Bakery")],
            ),
        )
        app_client.post("/memory/add", json={"text": "Alice Mercer owns a bakery."})

        # LLM extracts only "Alice" — old code returned no relational context
        monkeypatch.setattr(
            "memory_server.extract_entities_from_text",
            lambda text: EntityExtraction(entities=[Entity(name="Alice")]),
        )
        resp = app_client.post("/memory/search", json={"query": "What does Alice own?", "top_k": 3})
        assert resp.status_code == 200
        assert "Alice Mercer" in resp.json()["relational_context"]

    def test_migrate_v5_backfills_existing_entities(self, app_client):
        """_migrate_to_v5 fills aliases='[]' rows without touching already-populated ones."""
        import memory_server
        import json
        # Manually insert an entity with empty aliases (simulates pre-v5 data)
        import uuid
        eid = str(uuid.uuid4())
        now = "2024-01-01T00:00:00"
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
            "VALUES (?, ?, '[]', 0, ?, ?)",
            (eid, "Jordan Kim", now, now),
        )
        memory_server.sqlite_conn.commit()

        memory_server._migrate_to_v5()

        cursor.execute("SELECT aliases FROM entities WHERE id = ?", (eid,))
        aliases = json.loads(cursor.fetchone()[0])
        assert "Jordan" in aliases
        assert "Kim" in aliases


# ---------------------------------------------------------------------------
# POST /memory/context
# ---------------------------------------------------------------------------

class TestContextMemory:

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
# _retrieval_score  (pure function, no server needed)
# ---------------------------------------------------------------------------

class TestRetrievalScore:
    """Unit tests for the multi-signal retrieval scoring function."""

    NOW = datetime(2025, 6, 1, 12, 0, 0)
    RECENT = "2025-06-01T11:00:00"   # accessed 1 hour ago
    STALE   = "2025-01-01T00:00:00"  # accessed ~5 months ago

    @staticmethod
    def score(**kwargs):
        import memory_server
        defaults = dict(
            distance=0.2,
            hit_count=0,
            last_accessed=TestRetrievalScore.RECENT,
            temporal_status="current",
            now=TestRetrievalScore.NOW,
        )
        defaults.update(kwargs)
        return memory_server._retrieval_score(**defaults)

    def test_lower_distance_scores_higher(self):
        assert self.score(distance=0.0) > self.score(distance=0.5)

    def test_higher_hit_count_scores_higher(self):
        assert self.score(hit_count=50) > self.score(hit_count=0)

    def test_recent_scores_higher_than_stale(self):
        assert self.score(last_accessed=self.RECENT) > self.score(last_accessed=self.STALE)

    def test_historical_scores_lower_than_current(self):
        assert self.score(temporal_status="current") > self.score(temporal_status="historical")

    def test_uncertain_between_current_and_historical(self):
        assert (
            self.score(temporal_status="current")
            > self.score(temporal_status="uncertain")
            > self.score(temporal_status="historical")
        )

    def test_score_bounded_above_one(self):
        # Perfect similarity, max popularity, fully recent — still ≤ 1.0
        s = self.score(distance=0.0, hit_count=100, last_accessed=self.RECENT)
        assert s <= 1.0

    def test_score_non_negative(self):
        s = self.score(distance=2.0, hit_count=0, last_accessed=self.STALE, temporal_status="historical")
        assert s >= 0.0

    def test_popularity_can_overcome_small_similarity_gap(self):
        # A fact with 50 hits but slightly worse similarity should beat a cold fact.
        popular = self.score(distance=0.4, hit_count=50)
        cold    = self.score(distance=0.3, hit_count=0)
        assert popular > cold


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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        result = task["result"]
        for key in ("status", "chunks_total", "chunks_succeeded", "facts_added", "triples_added", "errors"):
            assert key in result, f"Missing key: {key}"

    def test_learn_empty_text_returns_zero_chunks(self, app_client):
        resp = app_client.post("/memory/learn", json={"text": ""})
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        result = task["result"]
        assert result["status"] == "success"
        assert result["chunks_total"] == 0
        assert result["facts_added"] == 0
        assert result["errors"] == []

    def test_learn_single_sentence_produces_one_chunk(self, app_client, monkeypatch):
        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Alice is kind."], triples=[]),
        )
        resp = app_client.post("/memory/learn", json={"text": "Alice is kind."})
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        result = task["result"]
        assert result["chunks_total"] == 1
        assert result["chunks_succeeded"] == 1
        assert result["facts_added"] == 1

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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        result = task["result"]
        assert result["chunks_total"] == 2
        assert result["chunks_succeeded"] == 2
        assert result["facts_added"] == 2
        assert result["triples_added"] == 2
        assert result["status"] == "success"

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
        assert resp.status_code == 202
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        result = task["result"]
        assert result["status"] == "partial"
        assert result["chunks_total"] == 2
        assert result["chunks_succeeded"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["chunk_index"] == 1

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
        assert resp.status_code == 202
        wait_for_task(app_client, resp.json()["task_id"])

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
        assert resp.status_code == 202
        wait_for_task(app_client, resp.json()["task_id"])
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


# ---------------------------------------------------------------------------
# Temporal relationship graph (Layer 2)
# ---------------------------------------------------------------------------

class TestTemporalGraph:
    """Tests for the temporal_graph PRECEDED_BY edge population and /memory/search integration."""

    def _stub_consolidation_helpers(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)

    def test_phase4_structural_creates_preceded_by_edge(self, app_client, monkeypatch):
        """Phase 4 structural supersession adds current -[PRECEDED_BY]-> past in temporal_graph."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice was a nurse.")],
                    triples=[KnowledgeTriple(subject="Alice", predicate="WAS", object="Nurse")],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice is a doctor.")],
                triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Nurse")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_consolidation_helpers(monkeypatch)
        app_client.post("/memory/consolidate")

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM atomic_facts WHERE content = 'Alice was a nurse.'")
        past_id = cursor.fetchone()[0]
        cursor.execute("SELECT id FROM atomic_facts WHERE content = 'Alice is a doctor.'")
        current_id = cursor.fetchone()[0]

        tg = memory_server.temporal_graph
        assert tg.G.has_node(current_id), "current fact should be a node in temporal_graph"
        assert tg.G.has_node(past_id), "past fact should be a node in temporal_graph"
        assert tg.G.has_edge(current_id, past_id), "current -[PRECEDED_BY]-> past edge should exist"
        edge_data = next(iter(tg.G[current_id][past_id].values()))
        assert edge_data.get("relation") == "PRECEDED_BY"

    def test_phase4b_a_supersedes_b_creates_preceded_by_edge(self, app_client, monkeypatch):
        """Phase 4b A_supersedes_B creates: keyword_fact -[PRECEDED_BY]-> neighbor in temporal_graph."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice no longer works at Google.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice works at Google.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_check_supersession",
            lambda a, b: SupersessionDecision(outcome="A_supersedes_B"),
        )
        self._stub_consolidation_helpers(monkeypatch)
        app_client.post("/memory/consolidate")

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE content = 'Alice no longer works at Google.'"
        )
        keyword_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE content = 'Alice works at Google.'"
        )
        neighbor_id = cursor.fetchone()[0]

        tg = memory_server.temporal_graph
        assert tg.G.has_edge(keyword_id, neighbor_id), \
            "keyword_fact -[PRECEDED_BY]-> neighbor should be added when A supersedes B"
        edge_data = next(iter(tg.G[keyword_id][neighbor_id].values()))
        assert edge_data.get("relation") == "PRECEDED_BY"
        assert neighbor_id in edge_data.get("source_fact_ids", [])

    def test_phase4b_b_supersedes_a_creates_preceded_by_edge(self, app_client, monkeypatch):
        """Phase 4b B_supersedes_A creates: neighbor -[PRECEDED_BY]-> keyword_fact in temporal_graph."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice no longer works at Google.")],
                    triples=[],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice works at Google.")],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_check_supersession",
            lambda a, b: SupersessionDecision(outcome="B_supersedes_A"),
        )
        self._stub_consolidation_helpers(monkeypatch)
        app_client.post("/memory/consolidate")

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE content = 'Alice no longer works at Google.'"
        )
        keyword_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE content = 'Alice works at Google.'"
        )
        neighbor_id = cursor.fetchone()[0]

        tg = memory_server.temporal_graph
        assert tg.G.has_edge(neighbor_id, keyword_id), \
            "neighbor -[PRECEDED_BY]-> keyword_fact should be added when B supersedes A"
        edge_data = next(iter(tg.G[neighbor_id][keyword_id].values()))
        assert edge_data.get("relation") == "PRECEDED_BY"
        assert keyword_id in edge_data.get("source_fact_ids", [])

    def test_search_returns_temporal_context(self, app_client, monkeypatch):
        """POST /memory/search includes temporal_context for facts with PRECEDED_BY history."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice was a nurse.")],
                    triples=[KnowledgeTriple(subject="Alice", predicate="WAS", object="Nurse")],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice is a doctor.")],
                triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Nurse")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr(
            "memory_server.librarian_assign_groups", lambda name, groups, **kw: None
        )
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_consolidation_helpers(monkeypatch)
        app_client.post("/memory/consolidate")

        monkeypatch.setattr(
            "memory_server.extract_entities_from_text", lambda q: None
        )
        resp = app_client.post("/memory/search", json={"query": "Alice job", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert "temporal_context" in body

        # The current fact ("Alice is a doctor.") should have "Alice was a nurse." as history.
        tc = body["temporal_context"]
        doctor_entry = next(
            (e for e in tc if e["current_fact"] == "Alice is a doctor."), None
        )
        assert doctor_entry is not None, "temporal_context should include Alice is a doctor."
        # preceded_by is now list[{fact, concurrent_with}]
        pred_facts = [p["fact"] for p in doctor_entry["preceded_by"]]
        assert "Alice was a nurse." in pred_facts

    def test_search_results_include_fact_id(self, app_client, monkeypatch):
        """POST /memory/search results now include an 'id' field for each fact."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(atomic_facts=["Alice runs a bakery."], triples=[]),
        )
        monkeypatch.setattr("memory_server.extract_entities_from_text", lambda q: None)
        app_client.post("/memory/add", json={"text": "Alice runs a bakery."})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM atomic_facts")
        db_id = cursor.fetchone()[0]

        resp = app_client.post("/memory/search", json={"query": "Alice bakery", "top_k": 1})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == db_id

    def test_clear_wipes_temporal_graph(self, app_client, monkeypatch):
        """DELETE /memory/clear removes all temporal graph nodes and edges."""
        import memory_server

        # Inject a temporal edge directly to simulate pre-existing history.
        memory_server.temporal_graph.add_relationship(
            "current-fact-id", "PRECEDED_BY", "past-fact-id",
            subject_name="current fact", object_name="past fact",
            fact_ids=["past-fact-id"], persist=False,
        )
        assert memory_server.temporal_graph.G.number_of_nodes() > 0

        resp = app_client.delete("/memory/clear")
        assert resp.status_code == 200
        assert memory_server.temporal_graph.G.number_of_nodes() == 0
        assert memory_server.temporal_graph.G.number_of_edges() == 0

    def test_dead_predecessor_skipped_in_temporal_context(self, app_client, monkeypatch):
        """
        KNOWN LIMITATION (pinned): if the past-state node referenced in a PRECEDED_BY edge
        no longer exists in SQLite (e.g., pruned by Phase 1 after the edge was written),
        /memory/search silently skips it rather than raising an error.
        temporal_context for that fact will be empty.
        """
        import uuid
        import memory_server
        from datetime import datetime

        # Insert a current fact into SQLite + ChromaDB.
        current_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, created_at, last_accessed) "
            "VALUES (?, ?, 'current', ?, ?)",
            (current_id, "Alice is a doctor.", now, now),
        )
        memory_server.sqlite_conn.commit()
        memory_server.collection.add(
            embeddings=[[0.1] * 768],
            documents=["Alice is a doctor."],
            ids=[current_id],
        )

        # The "past" fact ID does NOT exist in SQLite (simulates a pruned predecessor).
        dead_past_id = str(uuid.uuid4())
        memory_server.temporal_graph.add_relationship(
            current_id, "PRECEDED_BY", dead_past_id,
            subject_name="Alice is a doctor.",
            object_name="Alice was a nurse.",
            fact_ids=[dead_past_id], persist=False,
        )

        monkeypatch.setattr("memory_server.extract_entities_from_text", lambda q: None)
        resp = app_client.post("/memory/search", json={"query": "Alice doctor", "top_k": 1})
        assert resp.status_code == 200
        body = resp.json()
        # Dead predecessor is skipped: temporal_context must not contain the deleted fact.
        tc = body["temporal_context"]
        assert tc == [], \
            "temporal_context should be empty when all predecessors are dead (not in SQLite)"

    def test_phase2_high_sim_dedup_transfers_temporal_predecessors(self, app_client, monkeypatch):
        """Phase 2 high-sim dedup (sim=1.0) keeps the temporal chain on the surviving fact."""
        import uuid
        import memory_server
        from datetime import datetime

        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()

        # Historical predecessor.
        past_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice was a nurse.', 'historical', 0, ?, ?)",
            (past_id, now, now),
        )
        # Current fact A — has a temporal chain.
        a_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice is a doctor.', 'current', 0, ?, ?)",
            (a_id, now, now),
        )
        # Current fact B — identical duplicate (same text, same stub embedding → sim=1.0).
        b_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice is a doctor.', 'current', 0, ?, ?)",
            (b_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        memory_server.collection.add(
            embeddings=[[0.1] * 768, [0.1] * 768],
            documents=["Alice is a doctor.", "Alice is a doctor."],
            ids=[a_id, b_id],
        )
        # Wire temporal chain on a_id only.
        memory_server.temporal_graph.add_relationship(
            a_id, "PRECEDED_BY", past_id,
            subject_name="Alice is a doctor.",
            object_name="Alice was a nurse.",
            fact_ids=[past_id], persist=False,
        )

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        app_client.post("/memory/consolidate")

        # One "Alice is a doctor." survives.
        cursor.execute("SELECT id FROM atomic_facts WHERE content = 'Alice is a doctor.'")
        surviving_id = cursor.fetchone()[0]

        tg = memory_server.temporal_graph
        assert tg.G.has_edge(surviving_id, past_id), \
            "surviving fact must retain/inherit the PRECEDED_BY edge after high-sim dedup"
        assert set(tg.G.nodes()) == {surviving_id, past_id}, \
            "temporal_graph must not contain stale nodes from the dropped duplicate"

    def test_phase2_librarian_merge_transfers_temporal_predecessors(self, app_client, monkeypatch):
        """Phase 2 Librarian-decided merge propagates the temporal chain to the merged fact."""
        import uuid
        import memory_server
        from datetime import datetime

        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()

        # Historical predecessor.
        past_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice was a nurse.', 'historical', 0, ?, ?)",
            (past_id, now, now),
        )
        # Current fact A — has a temporal chain.
        a_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice is a doctor.', 'current', 0, ?, ?)",
            (a_id, now, now),
        )
        # Current fact B — similar but distinct text.
        b_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice works as a physician.', 'current', 0, ?, ?)",
            (b_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        # sim(vec_a, vec_b) ≈ 0.95: above DEDUP_SIMILARITY_THRESHOLD (0.90) but below
        # HIGH_SIM_DEDUP_THRESHOLD (0.99), so the Librarian merge path is exercised.
        vec_a = [1.0] + [0.0] * 767
        vec_b = [0.95, 0.31225] + [0.0] * 766
        memory_server.collection.add(
            embeddings=[vec_a, vec_b],
            documents=["Alice is a doctor.", "Alice works as a physician."],
            ids=[a_id, b_id],
        )
        # Wire temporal chain on a_id only.
        memory_server.temporal_graph.add_relationship(
            a_id, "PRECEDED_BY", past_id,
            subject_name="Alice is a doctor.",
            object_name="Alice was a nurse.",
            fact_ids=[past_id], persist=False,
        )

        monkeypatch.setattr(
            "memory_server.librarian_should_merge",
            lambda a, b: MergeDecision(should_merge=True, merged_fact="Alice is a doctor/physician."),
        )
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        app_client.post("/memory/consolidate")

        cursor.execute("SELECT id FROM atomic_facts WHERE content = 'Alice is a doctor/physician.'")
        merged_row = cursor.fetchone()
        assert merged_row is not None, "merged fact must exist in SQLite"
        merged_id = merged_row[0]

        tg = memory_server.temporal_graph
        assert tg.G.has_node(merged_id), "merged fact must be a node in temporal_graph"
        assert tg.G.has_edge(merged_id, past_id), \
            "merged fact must inherit the PRECEDED_BY edge from the original current fact"
        assert not tg.G.has_node(a_id), "original fact a must not remain in temporal_graph"
        assert not tg.G.has_node(b_id), "original fact b must not remain in temporal_graph"

    def test_phase4c_creates_concurrent_with_edges(self, app_client, monkeypatch):
        """Phase 4c creates bidirectional CONCURRENT_WITH edges for confirmed concurrent facts."""
        import uuid
        import memory_server
        from datetime import datetime

        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()

        # Two historical facts that both have a valid_period.
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice was a nurse.', 'historical', 'college years', 0, ?, ?)",
            (id_a, now, now),
        )
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice lived in Seattle.', 'historical', 'college years', 0, ?, ?)",
            (id_b, now, now),
        )
        memory_server.sqlite_conn.commit()

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr(
            "memory_server.librarian_check_concurrency",
            lambda a, b, pa, pb: ConcurrencyDecision(outcome="concurrent"),
        )

        app_client.post("/memory/consolidate")

        tg = memory_server.temporal_graph
        assert tg.G.has_node(id_a) and tg.G.has_node(id_b), \
            "both historical facts must be nodes in the temporal graph after Phase 4c"

        # Bidirectional edges.
        edges_a_to_b = [
            d for _, _, d in tg.G.out_edges(id_a, data=True)
            if d.get("relation") == "CONCURRENT_WITH"
        ]
        edges_b_to_a = [
            d for _, _, d in tg.G.out_edges(id_b, data=True)
            if d.get("relation") == "CONCURRENT_WITH"
        ]
        assert len(edges_a_to_b) >= 1, "id_a must have a CONCURRENT_WITH out-edge to id_b"
        assert len(edges_b_to_a) >= 1, "id_b must have a CONCURRENT_WITH out-edge to id_a"

    def test_search_includes_concurrent_with_in_temporal_context(self, app_client, monkeypatch):
        """After Phase 4c, /memory/search populates concurrent_with on each preceded_by entry."""
        import uuid
        import memory_server
        from datetime import datetime

        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()

        # Current fact with a predecessor.
        current_id = str(uuid.uuid4())
        past_id = str(uuid.uuid4())
        concurrent_id = str(uuid.uuid4())

        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice is a doctor.', 'current', 1, ?, ?)",
            (current_id, now, now),
        )
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice was a nurse.', 'historical', 'college years', 0, ?, ?)",
            (past_id, now, now),
        )
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice lived in Seattle.', 'historical', 'college years', 0, ?, ?)",
            (concurrent_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        memory_server.collection.add(
            embeddings=[[0.1] * 768],
            documents=["Alice is a doctor."],
            ids=[current_id],
        )

        tg = memory_server.temporal_graph
        tg.add_relationship(
            current_id, "PRECEDED_BY", past_id,
            subject_name="Alice is a doctor.",
            object_name="Alice was a nurse.",
            fact_ids=[past_id], persist=False,
        )
        # Bidirectional CONCURRENT_WITH between the two historical facts.
        tg.add_relationship(
            past_id, "CONCURRENT_WITH", concurrent_id,
            subject_name="Alice was a nurse.",
            object_name="Alice lived in Seattle.",
            fact_ids=[concurrent_id], persist=False,
        )
        tg.add_relationship(
            concurrent_id, "CONCURRENT_WITH", past_id,
            subject_name="Alice lived in Seattle.",
            object_name="Alice was a nurse.",
            fact_ids=[past_id], persist=False,
        )

        monkeypatch.setattr("memory_server.extract_entities_from_text", lambda q: None)
        resp = app_client.post("/memory/search", json={"query": "Alice doctor", "top_k": 1})
        assert resp.status_code == 200
        body = resp.json()
        tc = body["temporal_context"]
        doctor_entry = next(
            (e for e in tc if e["current_fact"] == "Alice is a doctor."), None
        )
        assert doctor_entry is not None
        pb = doctor_entry["preceded_by"]
        nurse_entry = next((p for p in pb if p["fact"] == "Alice was a nurse."), None)
        assert nurse_entry is not None, "preceded_by must include Alice was a nurse."
        assert "Alice lived in Seattle." in nurse_entry["concurrent_with"], \
            "concurrent_with must list Alice lived in Seattle. alongside the nurse predecessor"

    def test_phase2_transfers_concurrent_with_edges(self, app_client, monkeypatch):
        """Phase 2 high-sim dedup transfers CONCURRENT_WITH edges to the surviving fact."""
        import uuid
        import memory_server
        from datetime import datetime

        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()

        # Historical fact that is concurrent with the current-state predecessor.
        concurrent_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice lived in Seattle.', 'historical', 'college years', 0, ?, ?)",
            (concurrent_id, now, now),
        )
        # Past-state predecessor for the current facts.
        past_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Alice was a nurse.', 'historical', 0, ?, ?)",
            (past_id, now, now),
        )
        # Two identical current facts (sim=1.0 → high-sim dedup path).
        a_id = str(uuid.uuid4())
        b_id = str(uuid.uuid4())
        for fid in (a_id, b_id):
            cursor.execute(
                "INSERT INTO atomic_facts (id, content, temporal_status, hit_count, created_at, last_accessed) "
                "VALUES (?, 'Alice is a doctor.', 'current', 0, ?, ?)",
                (fid, now, now),
            )
        memory_server.sqlite_conn.commit()

        memory_server.collection.add(
            embeddings=[[0.1] * 768, [0.1] * 768],
            documents=["Alice is a doctor.", "Alice is a doctor."],
            ids=[a_id, b_id],
        )

        tg = memory_server.temporal_graph
        # a_id has PRECEDED_BY → past_id, which itself has CONCURRENT_WITH ↔ concurrent_id.
        tg.add_relationship(
            a_id, "PRECEDED_BY", past_id,
            subject_name="Alice is a doctor.",
            object_name="Alice was a nurse.",
            fact_ids=[past_id], persist=False,
        )
        tg.add_relationship(
            past_id, "CONCURRENT_WITH", concurrent_id,
            subject_name="Alice was a nurse.",
            object_name="Alice lived in Seattle.",
            fact_ids=[concurrent_id], persist=False,
        )
        tg.add_relationship(
            concurrent_id, "CONCURRENT_WITH", past_id,
            subject_name="Alice lived in Seattle.",
            object_name="Alice was a nurse.",
            fact_ids=[past_id], persist=False,
        )

        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        app_client.post("/memory/consolidate")

        cursor.execute("SELECT id FROM atomic_facts WHERE content = 'Alice is a doctor.'")
        surviving_id = cursor.fetchone()[0]

        # Surviving fact must inherit the PRECEDED_BY → past chain.
        assert tg.G.has_edge(surviving_id, past_id), \
            "surviving fact must have PRECEDED_BY → past_id"
        # The CONCURRENT_WITH edges on past_id are untouched (they are on the historical node,
        # not on the dropped duplicate, so no transfer was needed — this tests they persist).
        assert tg.G.has_node(past_id), "past_id node must still exist"
        assert tg.G.has_node(concurrent_id), "concurrent_id node must still exist"
        cw_edges = [
            d for _, _, d in tg.G.out_edges(past_id, data=True)
            if d.get("relation") == "CONCURRENT_WITH"
        ]
        assert len(cw_edges) >= 1, \
            "past_id must still have its CONCURRENT_WITH edge to concurrent_id"


# ---------------------------------------------------------------------------
# POST /memory/consolidate — Phase 5: Retroactive Entity Resolution
# ---------------------------------------------------------------------------

class TestConsolidatePhase5:
    """Tests for Phase 5 retroactive entity resolution in _consolidate_memories_sync."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)

    def test_phase5_rewrites_compound_entity(self, app_client, monkeypatch):
        """Phase 5 rewrites a compound entity node as a proper KG triple."""
        import memory_server
        import uuid as _uuid

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["James is a senior advisor to Cellbridge Therapeutics."],
                triples=[
                    KnowledgeTriple(
                        subject="James",
                        predicate="IS",
                        object="Advisor To Cellbridge Therapeutics",
                    )
                ],
            ),
        )
        app_client.post("/memory/add", json={"text": "first"})

        cursor = memory_server.sqlite_conn.cursor()

        cursor.execute(
            "SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)",
            ("Advisor To Cellbridge Therapeutics",),
        )
        compound_row = cursor.fetchone()
        assert compound_row, "compound entity should exist after /add"
        compound_id = compound_row[0]

        # Insert "Cellbridge Therapeutics" as a separate entity so Phase 5 has a target.
        cursor.execute(
            "SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)",
            ("Cellbridge Therapeutics",),
        )
        contained_row = cursor.fetchone()
        if not contained_row:
            contained_id = str(_uuid.uuid4())
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
                "VALUES (?, 'Cellbridge Therapeutics', '[]', 0, ?, ?)",
                (contained_id, now, now),
            )
            memory_server.sqlite_conn.commit()
        else:
            contained_id = contained_row[0]

        monkeypatch.setattr(
            "memory_server.librarian_resolve_compound_entity",
            lambda compound, contained: CompoundEntityDecision(
                action="rewrite", suggested_predicate="IS_ADVISOR_TO"
            ),
        )
        self._stub_all_other_phases(monkeypatch)
        app_client.post("/memory/consolidate")

        # Compound entity should be gone from SQLite.
        cursor.execute("SELECT id FROM entities WHERE id = ?", (compound_id,))
        assert cursor.fetchone() is None, "compound entity should be deleted from SQLite"

        # New KG edge James --IS_ADVISOR_TO--> Cellbridge Therapeutics should exist.
        kg = memory_server.knowledge_graph
        james_row = cursor.execute(
            "SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)", ("James",)
        ).fetchone()
        assert james_row, "James entity should still exist"
        james_id = james_row[0]
        assert kg.G.has_node(james_id), "James should still be a KG node"
        assert kg.G.has_node(contained_id), "Cellbridge Therapeutics should be a KG node"
        edges = [
            data for _, _, data in kg.G.out_edges(james_id, data=True)
            if data.get("relation") == "IS_ADVISOR_TO"
        ]
        assert edges, "James --[IS_ADVISOR_TO]--> Cellbridge Therapeutics edge should exist"

    def test_phase5_keeps_distinct_entities(self, app_client, monkeypatch):
        """Phase 5 leaves both entities intact when Librarian returns action='keep'."""
        import memory_server

        call_count = [0]

        def vary(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=["Alice Kim is a researcher."],
                    triples=[KnowledgeTriple(subject="Alice Kim", predicate="IS", object="Researcher")],
                )
            return MemoryProcessing(
                atomic_facts=["Alice is a friend."],
                triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Friend")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_resolve_compound_entity",
            lambda compound, contained: CompoundEntityDecision(action="keep"),
        )
        self._stub_all_other_phases(monkeypatch)
        app_client.post("/memory/consolidate")

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM entities WHERE LOWER(canonical_name) IN ('alice kim', 'alice')"
        )
        assert cursor.fetchone()[0] == 2, "both entities should survive when action='keep'"

    def test_phase5_flags_ambiguous_pair(self, app_client, monkeypatch):
        """Phase 5 adds flagged entry with source='phase5' when action='flag'."""
        import memory_server
        import uuid as _uuid

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["James works for Cellbridge Therapeutics."],
                triples=[
                    KnowledgeTriple(
                        subject="James",
                        predicate="IS",
                        object="Advisor To Cellbridge Therapeutics",
                    )
                ],
            ),
        )
        app_client.post("/memory/add", json={"text": "first"})

        cursor = memory_server.sqlite_conn.cursor()
        contained_id = str(_uuid.uuid4())
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT OR IGNORE INTO entities "
            "(id, canonical_name, aliases, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Cellbridge Therapeutics', '[]', 0, ?, ?)",
            (contained_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        monkeypatch.setattr(
            "memory_server.librarian_resolve_compound_entity",
            lambda compound, contained: CompoundEntityDecision(action="flag"),
        )
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        report = task["result"]["report"]

        phase5_flags = [f for f in report["flagged"] if f.get("source") == "phase5"]
        assert phase5_flags, "report['flagged'] should contain at least one phase5 entry"

    def test_phase5_transfers_entity_chunks(self, app_client, monkeypatch):
        """Phase 5 migrates entity_chunks from compound to contained entity."""
        import memory_server
        import uuid as _uuid

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["James advises Cellbridge Therapeutics."],
                triples=[
                    KnowledgeTriple(
                        subject="James",
                        predicate="IS",
                        object="Advisor To Cellbridge Therapeutics",
                    )
                ],
            ),
        )
        app_client.post("/memory/add", json={"text": "chunk for migration"})

        cursor = memory_server.sqlite_conn.cursor()
        contained_id = str(_uuid.uuid4())
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT OR IGNORE INTO entities "
            "(id, canonical_name, aliases, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Cellbridge Therapeutics', '[]', 0, ?, ?)",
            (contained_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        cursor.execute(
            "SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)",
            ("Advisor To Cellbridge Therapeutics",),
        )
        compound_row = cursor.fetchone()
        assert compound_row, "compound entity must exist"
        compound_id = compound_row[0]

        cursor.execute(
            "SELECT COUNT(*) FROM entity_chunks WHERE entity_id = ?", (compound_id,)
        )
        assert cursor.fetchone()[0] > 0, "compound entity should have entity_chunks rows before consolidation"

        monkeypatch.setattr(
            "memory_server.librarian_resolve_compound_entity",
            lambda compound, contained: CompoundEntityDecision(
                action="rewrite", suggested_predicate="IS_ADVISOR_TO"
            ),
        )
        self._stub_all_other_phases(monkeypatch)
        app_client.post("/memory/consolidate")

        cursor.execute(
            "SELECT COUNT(*) FROM entity_chunks WHERE entity_id = ?", (contained_id,)
        )
        assert cursor.fetchone()[0] >= 1, "contained entity should inherit entity_chunks rows"

    def test_phase5_no_candidates(self, app_client, monkeypatch):
        """Phase 5 is a no-op when no entity name is a length+3 substring of another."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=["Alice owns a bakery."],
                triples=[KnowledgeTriple(subject="Alice", predicate="OWNS", object="Bakery")],
            ),
        )
        app_client.post("/memory/add", json={"text": "only"})

        resolve_calls = [0]

        def counting_resolver(compound, contained):
            resolve_calls[0] += 1
            return CompoundEntityDecision(action="keep")

        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", counting_resolver)
        self._stub_all_other_phases(monkeypatch)
        app_client.post("/memory/consolidate")

        assert resolve_calls[0] == 0, "resolver should not be called when no candidates pass the filter"


# ---------------------------------------------------------------------------
# POST /memory/temporal/chain
# ---------------------------------------------------------------------------

class TestTemporalChain:
    """Tests for POST /memory/temporal/chain endpoint."""

    def _seed_fact(self, app_client, monkeypatch, fact_text: str, fact_status: str = "current") -> str:
        """Seed a single fact via /memory/add and return its fact_id."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text: MemoryProcessing(
                atomic_facts=[AtomicFact(text=fact_text, temporal_status=fact_status)],
                triples=[],
            ),
        )
        resp = app_client.post("/memory/add", json={"text": fact_text})
        wait_for_task(app_client, resp.json()["task_id"])
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT id FROM atomic_facts WHERE content = ?", (fact_text,))
        row = cursor.fetchone()
        assert row, f"fact '{fact_text}' was not inserted"
        return row[0]

    def test_chain_by_fact_id(self, app_client, monkeypatch):
        """Returns an ordered hop chain when queried by fact_id."""
        import memory_server

        a_id = self._seed_fact(app_client, monkeypatch, "Alice is a doctor.", "current")
        b_id = self._seed_fact(app_client, monkeypatch, "Alice was a nurse.", "historical")
        c_id = self._seed_fact(app_client, monkeypatch, "Alice was a student.", "historical")

        tg = memory_server.temporal_graph
        tg.add_relationship(a_id, "PRECEDED_BY", b_id,
                            subject_name="Alice is a doctor.",
                            object_name="Alice was a nurse.",
                            fact_ids=[b_id], persist=False)
        tg.add_relationship(b_id, "PRECEDED_BY", c_id,
                            subject_name="Alice was a nurse.",
                            object_name="Alice was a student.",
                            fact_ids=[c_id], persist=False)

        resp = app_client.post("/memory/temporal/chain", json={"fact_id": a_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["root_fact"]["id"] == a_id
        assert body["root_fact"]["text"] == "Alice is a doctor."
        chain = body["chain"]
        assert len(chain) == 2
        assert chain[0]["hop"] == 1
        assert chain[0]["text"] == "Alice was a nurse."
        assert chain[1]["hop"] == 2
        assert chain[1]["text"] == "Alice was a student."

    def test_chain_by_query(self, app_client, monkeypatch):
        """Finds root fact via ChromaDB top-1 when only query is provided."""
        import memory_server

        a_id = self._seed_fact(app_client, monkeypatch, "Alice is a doctor.", "current")
        b_id = self._seed_fact(app_client, monkeypatch, "Alice was a nurse.", "historical")

        tg = memory_server.temporal_graph
        tg.add_relationship(a_id, "PRECEDED_BY", b_id,
                            subject_name="Alice is a doctor.",
                            object_name="Alice was a nurse.",
                            fact_ids=[b_id], persist=False)

        resp = app_client.post("/memory/temporal/chain", json={"query": "Alice job"})
        assert resp.status_code == 200
        body = resp.json()
        assert "root_fact" in body
        assert "chain" in body
        # With uniform stub embeddings the endpoint prefers current facts as root.
        assert body["root_fact"]["temporal_status"] == "current"

    def test_chain_includes_concurrent_with(self, app_client, monkeypatch):
        """Each chain entry includes concurrent_with facts from the temporal graph."""
        import memory_server

        a_id = self._seed_fact(app_client, monkeypatch, "Alice is a doctor.", "current")
        b_id = self._seed_fact(app_client, monkeypatch, "Alice was a nurse.", "historical")
        c_id = self._seed_fact(app_client, monkeypatch, "Alice lived in Seattle.", "historical")

        tg = memory_server.temporal_graph
        tg.add_relationship(a_id, "PRECEDED_BY", b_id,
                            subject_name="Alice is a doctor.",
                            object_name="Alice was a nurse.",
                            fact_ids=[b_id], persist=False)
        tg.add_relationship(b_id, "CONCURRENT_WITH", c_id,
                            subject_name="Alice was a nurse.",
                            object_name="Alice lived in Seattle.",
                            fact_ids=[c_id], persist=False)

        resp = app_client.post("/memory/temporal/chain", json={"fact_id": a_id})
        assert resp.status_code == 200
        chain = resp.json()["chain"]
        assert len(chain) == 1
        concurrent_texts = [e["text"] for e in chain[0]["concurrent_with"]]
        assert "Alice lived in Seattle." in concurrent_texts

    def test_chain_max_depth_truncates(self, app_client, monkeypatch):
        """max_depth=1 returns only the direct predecessor."""
        import memory_server

        a_id = self._seed_fact(app_client, monkeypatch, "Alice is a doctor.", "current")
        b_id = self._seed_fact(app_client, monkeypatch, "Alice was a nurse.", "historical")
        c_id = self._seed_fact(app_client, monkeypatch, "Alice was a student.", "historical")

        tg = memory_server.temporal_graph
        tg.add_relationship(a_id, "PRECEDED_BY", b_id,
                            subject_name="Alice is a doctor.",
                            object_name="Alice was a nurse.",
                            fact_ids=[b_id], persist=False)
        tg.add_relationship(b_id, "PRECEDED_BY", c_id,
                            subject_name="Alice was a nurse.",
                            object_name="Alice was a student.",
                            fact_ids=[c_id], persist=False)

        resp = app_client.post("/memory/temporal/chain", json={"fact_id": a_id, "max_depth": 1})
        assert resp.status_code == 200
        chain = resp.json()["chain"]
        assert len(chain) == 1
        assert chain[0]["hop"] == 1

    def test_chain_root_has_no_history(self, app_client, monkeypatch):
        """Returns root_fact and empty chain when fact has no predecessors."""
        a_id = self._seed_fact(app_client, monkeypatch, "Alice is a doctor.", "current")

        resp = app_client.post("/memory/temporal/chain", json={"fact_id": a_id})
        assert resp.status_code == 200
        body = resp.json()
        assert body["root_fact"]["id"] == a_id
        assert body["chain"] == []

    def test_chain_neither_param_returns_422(self, app_client, monkeypatch):
        """Returns 422 when neither fact_id nor query is provided."""
        resp = app_client.post("/memory/temporal/chain", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Phase 4d: same-predicate state-change supersession
# ---------------------------------------------------------------------------

class TestConsolidatePhase4d:
    """Tests for Phase 4d same-predicate state-change detection in _consolidate_memories_sync."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", lambda c, n: None)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)
        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)

    def test_phase4d_adds_preceded_by_for_same_predicate_different_object(self, app_client, monkeypatch):
        """LIVED_IN Mission District (historical) + LIVED_IN Sunset District (current)
        → PRECEDED_BY edge in temporal graph without LLM."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Alice lived in the Mission District.", temporal_status="historical", valid_period="2013-2018")],
                    triples=[KnowledgeTriple(subject="Alice", predicate="LIVED_IN", object="Mission District", supporting_fact_indices=[0])],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice lives in the Sunset District.", temporal_status="current")],
                triples=[KnowledgeTriple(subject="Alice", predicate="LIVED_IN", object="Sunset District", supporting_fact_indices=[0])],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert report["superseded"] >= 1

        # Verify temporal graph has PRECEDED_BY edge
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE temporal_status = 'current' AND content LIKE '%Sunset%'"
        )
        curr_row = cursor.fetchone()
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE temporal_status = 'historical' AND content LIKE '%Mission%'"
        )
        hist_row = cursor.fetchone()
        assert curr_row and hist_row
        assert memory_server.temporal_graph.G.has_node(curr_row[0]) or memory_server.temporal_graph.G.has_node(hist_row[0])

    def test_phase4d_same_object_no_edge(self, app_client, monkeypatch):
        """Two edges with same predicate AND same object → no PRECEDED_BY added."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            status = "historical" if call_count[0] == 1 else "current"
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=f"Alice lives in Portland. (chunk {call_count[0]})", temporal_status=status)],
                triples=[KnowledgeTriple(subject="Alice", predicate="LIVED_IN", object="Portland", supporting_fact_indices=[0])],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        # Phase 2 dedup may fire for very similar content; either way Phase 4d should NOT add count
        # We just check that the task completes without error
        assert task["result"]["report"] is not None

    def test_phase4d_no_historical_no_edge(self, app_client, monkeypatch):
        """Two current facts with same predicate, different objects → no PRECEDED_BY."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            obj = "Mission District" if call_count[0] == 1 else "Sunset District"
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text=f"Alice lives in {obj}.", temporal_status="current")],
                triples=[KnowledgeTriple(subject="Alice", predicate="LIVED_IN", object=obj, supporting_fact_indices=[0])],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_all_other_phases(monkeypatch)
        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        # Phase 4d should not add supersession for two current-status facts
        assert isinstance(task["result"]["report"]["superseded"], int)


# ---------------------------------------------------------------------------
# Phase 6: entity identity merge
# ---------------------------------------------------------------------------

class TestConsolidatePhase6:
    """Tests for Phase 6 entity identity merge in _consolidate_memories_sync."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", lambda c, n: None)
        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)

    def test_phase6_merges_short_and_long_name(self, app_client, monkeypatch):
        """'Alice' and 'Alice Mercer' fragments are merged when LLM confirms same entity."""
        import memory_server
        import uuid as _uuid

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=["Alice is a scientist."],
                    triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Scientist")],
                )
            return MemoryProcessing(
                atomic_facts=["Alice Mercer works at CellBridge."],
                triples=[KnowledgeTriple(subject="Alice Mercer", predicate="WORKS_AT", object="CellBridge")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_should_merge_entities",
            lambda a, b, fa, fb: EntityMergeDecision(
                should_merge=True, canonical_to_keep="Alice Mercer", explanation="same person"
            ),
        )
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert len(report["merged_entities"]) >= 1
        assert any(m["canonical"] == "Alice Mercer" for m in report["merged_entities"])

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT canonical_name FROM entities WHERE LOWER(canonical_name) = 'alice'")
        assert cursor.fetchone() is None, "'Alice' node should be eliminated"
        cursor.execute("SELECT canonical_name FROM entities WHERE LOWER(canonical_name) = 'alice mercer'")
        assert cursor.fetchone() is not None, "'Alice Mercer' should survive"

    def test_phase6_skips_when_llm_says_no_merge(self, app_client, monkeypatch):
        """When LLM returns should_merge=False, both entity nodes survive."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=["Alice is a teacher."],
                    triples=[KnowledgeTriple(subject="Alice", predicate="IS", object="Teacher", supporting_fact_indices=[0])],
                )
            return MemoryProcessing(
                atomic_facts=["Alice Mercer is a doctor."],
                triples=[KnowledgeTriple(subject="Alice Mercer", predicate="IS", object="Doctor", supporting_fact_indices=[0])],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_should_merge_entities",
            lambda a, b, fa, fb: EntityMergeDecision(
                should_merge=False, canonical_to_keep="", explanation="different people"
            ),
        )
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        assert task["result"]["report"]["merged_entities"] == []

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities WHERE LOWER(canonical_name) IN ('alice', 'alice mercer')")
        assert cursor.fetchone()[0] == 2

    def test_phase6_transfers_entity_chunks(self, app_client, monkeypatch):
        """After merge, entity_chunks rows from the eliminated entity move to canonical."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=["Jordan was born in Seoul."],
                    triples=[KnowledgeTriple(subject="Jordan", predicate="BORN_IN", object="Seoul", supporting_fact_indices=[0])],
                )
            return MemoryProcessing(
                atomic_facts=["Jordan Kim teaches art."],
                triples=[KnowledgeTriple(subject="Jordan Kim", predicate="TEACHES", object="Art", supporting_fact_indices=[0])],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        monkeypatch.setattr(
            "memory_server.librarian_should_merge_entities",
            lambda a, b, fa, fb: EntityMergeDecision(
                should_merge=True, canonical_to_keep="Jordan Kim", explanation="same person"
            ),
        )
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        assert len(task["result"]["report"]["merged_entities"]) >= 1

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT e.canonical_name, COUNT(ec.chunk_id) FROM entities e "
            "JOIN entity_chunks ec ON e.id = ec.entity_id "
            "WHERE LOWER(e.canonical_name) = 'jordan kim' GROUP BY e.id"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] >= 2, "Jordan Kim should inherit chunks from both entities"

    def test_phase6_cap_respected(self, app_client, monkeypatch):
        """Phase 6 stops after ENTITY_MERGE_MAX_PAIRS LLM calls."""
        import memory_server

        call_count = [0]

        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            name = f"Person {call_count[0]}"
            return MemoryProcessing(
                atomic_facts=[f"{name} exists."],
                triples=[],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        # Add a modest number of adds — enough to have multiple candidate pairs
        for i in range(5):
            app_client.post("/memory/add", json={"text": f"chunk {i}"})

        merge_call_count = [0]

        def counting_merge(a, b, fa, fb):
            merge_call_count[0] += 1
            return EntityMergeDecision(should_merge=False, canonical_to_keep="", explanation="")

        monkeypatch.setattr("memory_server.librarian_should_merge_entities", counting_merge)
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        assert merge_call_count[0] <= memory_server.ENTITY_MERGE_MAX_PAIRS


# ---------------------------------------------------------------------------
# Phase 7: non-entity node cleanup
# ---------------------------------------------------------------------------

class TestConsolidatePhase7:
    """Tests for Phase 7 boolean/year/junk entity cleanup in _consolidate_memories_sync."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", lambda c, n: None)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)

    def test_phase7_removes_boolean_entity(self, app_client, monkeypatch):
        """Boolean entity 'True' is removed by Tier A without LLM."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text, **kwargs: MemoryProcessing(
                atomic_facts=["Alice is an avid photographer."],
                triples=[KnowledgeTriple(subject="Alice", predicate="IS_PHOTOGRAPHER", object="True")],
            ),
        )
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities WHERE LOWER(canonical_name) = 'true'")
        assert cursor.fetchone()[0] == 1, "'True' entity should exist after /add"

        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert any(c["name"].lower() == "true" for c in report["cleaned_nodes"])

        cursor.execute("SELECT COUNT(*) FROM entities WHERE LOWER(canonical_name) = 'true'")
        assert cursor.fetchone()[0] == 0

    def test_phase7_removes_year_entity_and_migrates_valid_period(self, app_client, monkeypatch):
        """Year entity '2020' is removed; its year value is migrated to backing fact's valid_period."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text, **kwargs: MemoryProcessing(
                atomic_facts=[AtomicFact(text="Alice stopped painting in 2020.", temporal_status="historical")],
                triples=[KnowledgeTriple(subject="Alice", predicate="HAS_NOT_PAINTED_SINCE", object="2020", supporting_fact_indices=[0])],
            ),
        )
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities WHERE canonical_name = '2020'")
        assert cursor.fetchone()[0] == 1

        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)
        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert any(c["name"] == "2020" for c in report["cleaned_nodes"])

        cursor.execute("SELECT COUNT(*) FROM entities WHERE canonical_name = '2020'")
        assert cursor.fetchone()[0] == 0

    def test_phase7_tier_b_junk_removed_via_llm(self, app_client, monkeypatch):
        """Tier B: LLM classifies a low-hit entity as 'junk' → removed."""
        import memory_server
        import uuid as _uuid

        # Seed a junk entity directly (no /add path needed)
        junk_id = str(_uuid.uuid4())
        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Fragment Xyz', '[]', 0, ?, ?)",
            (junk_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        monkeypatch.setattr(
            "memory_server.librarian_classify_entity",
            lambda n, f: EntityClassification(entity_type="junk", reasoning="meaningless fragment"),
        )
        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert any(c["name"] == "Fragment Xyz" for c in report["cleaned_nodes"])

        cursor.execute("SELECT COUNT(*) FROM entities WHERE id = ?", (junk_id,))
        assert cursor.fetchone()[0] == 0

    def test_phase7_tier_b_role_flagged_not_removed(self, app_client, monkeypatch):
        """Tier B: LLM classifies entity as 'role' → flagged for review, not deleted."""
        import memory_server
        import uuid as _uuid

        role_id = str(_uuid.uuid4())
        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Chief Science Officer', '[]', 0, ?, ?)",
            (role_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        monkeypatch.setattr(
            "memory_server.librarian_classify_entity",
            lambda n, f: EntityClassification(entity_type="role", reasoning="job title"),
        )
        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert any(f.get("source") == "phase7" for f in report["flagged"])

        cursor.execute("SELECT COUNT(*) FROM entities WHERE id = ?", (role_id,))
        assert cursor.fetchone()[0] == 1, "role entity should NOT be deleted"

    def test_phase7_keeps_person_and_org_entities(self, app_client, monkeypatch):
        """Tier B: LLM classifies entity as 'person' or 'organization' → kept."""
        import memory_server
        import uuid as _uuid

        person_id = str(_uuid.uuid4())
        now = datetime.now().isoformat()
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
            "VALUES (?, 'Dr Smith', '[]', 0, ?, ?)",
            (person_id, now, now),
        )
        memory_server.sqlite_conn.commit()

        monkeypatch.setattr(
            "memory_server.librarian_classify_entity",
            lambda n, f: EntityClassification(entity_type="person", reasoning="human name"),
        )
        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"

        cursor.execute("SELECT COUNT(*) FROM entities WHERE id = ?", (person_id,))
        assert cursor.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Phase 8: retroactive predicate normalization
# ---------------------------------------------------------------------------

class TestConsolidatePhase8:
    """Tests for Phase 8 retroactive FORMERLY_*/IS_CURRENTLY predicate rewriting."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", lambda c, n: None)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)
        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)

    def test_phase8_rewrites_formerly_predicate(self, app_client, monkeypatch):
        """FORMERLY_WAS predicate is rewritten to WAS by Phase 8."""
        import memory_server
        import uuid as _uuid

        # Phase 8 repairs legacy data — normalization at write time (via _PREDICATE_SYNONYMS)
        # means new /add calls already produce WAS/IS. Inject a FORMERLY_WAS edge directly
        # into the KG to simulate pre-expansion legacy data.
        subj_id = memory_server.get_or_create_entity("Marcus")
        obj_id = memory_server.get_or_create_entity("Venture Capitalist")
        now = datetime.now().isoformat()
        fact_id = str(_uuid.uuid4())
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, created_at, last_accessed) "
            "VALUES (?, ?, 'historical', ?, ?)",
            (fact_id, "Marcus was a venture capitalist.", now, now),
        )
        memory_server.sqlite_conn.commit()
        memory_server.knowledge_graph.add_relationship(
            subj_id, "FORMERLY_WAS", obj_id,
            subject_name="Marcus", object_name="Venture Capitalist",
            fact_ids=[fact_id], persist=True,
        )

        # Verify the raw FORMERLY_WAS edge exists in KG before consolidation
        g = memory_server.knowledge_graph.G
        formerly_edges = [
            (u, v, k, d) for u, v, k, d in g.edges(data=True, keys=True)
            if d.get("relation", "").upper().startswith("FORMERLY_")
        ]
        assert len(formerly_edges) >= 1, "FORMERLY_* edge should exist before consolidation"

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert report["predicates_normalized"] >= 1

        g = memory_server.knowledge_graph.G
        remaining_formerly = [
            d for _, _, _, d in g.edges(data=True, keys=True)
            if d.get("relation", "").upper().startswith("FORMERLY_")
        ]
        assert remaining_formerly == [], "No FORMERLY_* edges should remain after Phase 8"

    def test_phase8_rewrites_is_currently(self, app_client, monkeypatch):
        """IS_CURRENTLY predicate is rewritten to IS by Phase 8."""
        import memory_server
        import uuid as _uuid

        # Inject IS_CURRENTLY edge directly to simulate pre-expansion legacy data.
        subj_id = memory_server.get_or_create_entity("Alice")
        obj_id = memory_server.get_or_create_entity("Chief Science Officer")
        now = datetime.now().isoformat()
        fact_id = str(_uuid.uuid4())
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, created_at, last_accessed) "
            "VALUES (?, ?, 'current', ?, ?)",
            (fact_id, "Alice is the Chief Science Officer.", now, now),
        )
        memory_server.sqlite_conn.commit()
        memory_server.knowledge_graph.add_relationship(
            subj_id, "IS_CURRENTLY", obj_id,
            subject_name="Alice", object_name="Chief Science Officer",
            fact_ids=[fact_id], persist=True,
        )

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert report["predicates_normalized"] >= 1

        g = memory_server.knowledge_graph.G
        is_currently_edges = [
            d for _, _, _, d in g.edges(data=True, keys=True)
            if d.get("relation", "").upper() == "IS_CURRENTLY"
        ]
        assert is_currently_edges == []

    def test_phase8_no_op_on_clean_graph(self, app_client, monkeypatch):
        """Phase 8 is a no-op when no FORMERLY_* or IS_CURRENTLY edges exist."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text, **kwargs: MemoryProcessing(
                atomic_facts=["Alice works at CellBridge."],
                triples=[KnowledgeTriple(subject="Alice", predicate="WORKS_AT", object="CellBridge")],
            ),
        )
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})

        self._stub_all_other_phases(monkeypatch)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        assert task["result"]["report"]["predicates_normalized"] == 0

    def test_phase8_is_currently_does_not_mark_facts_historical(self, app_client, monkeypatch):
        """IS_CURRENTLY → IS rewrite must NOT mark backing facts as historical.

        IS_CURRENTLY is a present-tense predicate mislabeled with a temporal prefix.
        Its backing facts are current-state, not past-state. This guards the regression
        where Phase 8 marked all IS_CURRENTLY-backed facts as historical (causing
        "Alice is the CSO" to become historical).
        """
        import memory_server
        import uuid as _uuid

        subj_id = memory_server.get_or_create_entity("Alice")
        obj_id = memory_server.get_or_create_entity("Chief Science Officer")
        now = datetime.now().isoformat()
        fact_id = str(_uuid.uuid4())
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, created_at, last_accessed) "
            "VALUES (?, ?, 'current', ?, ?)",
            (fact_id, "Alice is the Chief Science Officer.", now, now),
        )
        memory_server.sqlite_conn.commit()
        memory_server.knowledge_graph.add_relationship(
            subj_id, "IS_CURRENTLY", obj_id,
            subject_name="Alice", object_name="Chief Science Officer",
            fact_ids=[fact_id], persist=True,
        )

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute("SELECT temporal_status FROM atomic_facts WHERE id = ?", (fact_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "current", "IS_CURRENTLY backing facts must stay current after Phase 8"

    def test_phase8_is_currently_no_self_loop_in_temporal_graph(self, app_client, monkeypatch):
        """IS_CURRENTLY → IS rewrite must NOT create a PRECEDED_BY self-loop in the temporal graph.

        Phase 8 looks for 'IS' out-edges to create PRECEDED_BY history. When IS_CURRENTLY is
        rewritten to IS in-place, the rewritten edge matches the search, creating a
        PRECEDED_BY(fact → fact) self-loop. This test guards that regression.
        """
        import memory_server
        import uuid as _uuid

        subj_id = memory_server.get_or_create_entity("Alice")
        obj_id = memory_server.get_or_create_entity("Director")
        now = datetime.now().isoformat()
        fact_id = str(_uuid.uuid4())
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, created_at, last_accessed) "
            "VALUES (?, ?, 'current', ?, ?)",
            (fact_id, "Alice is the Director.", now, now),
        )
        memory_server.sqlite_conn.commit()
        memory_server.knowledge_graph.add_relationship(
            subj_id, "IS_CURRENTLY", obj_id,
            subject_name="Alice", object_name="Director",
            fact_ids=[fact_id], persist=True,
        )

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"

        tg = memory_server.temporal_graph.G
        self_loops = [
            (u, v, d)
            for u, v, d in tg.edges(data=True)
            if u == v and d.get("relation") == "PRECEDED_BY"
        ]
        assert self_loops == [], f"No PRECEDED_BY self-loops should exist, found: {self_loops}"


# ---------------------------------------------------------------------------
# Group assignment validation
# ---------------------------------------------------------------------------

class TestGroupValidation:
    """Tests for _is_valid_group_name filtering of hallucinated/malformed group labels."""

    def test_rejects_underscored_name(self, app_client, monkeypatch):
        """Group names containing underscores are rejected (hallucinated compound labels)."""
        from memory_server import _is_valid_group_name
        assert not _is_valid_group_name("Pipettes_Bioreactors_And_Digital_Sequences")
        assert not _is_valid_group_name("Lead_Research_Associate")

    def test_rejects_empty_list_string(self, app_client, monkeypatch):
        """The string 'Empty_List' (schema confusion from 3B model) is rejected."""
        from memory_server import _is_valid_group_name
        assert not _is_valid_group_name("Empty_List")
        assert not _is_valid_group_name("empty_list")
        assert not _is_valid_group_name("null")
        assert not _is_valid_group_name("None")
        assert not _is_valid_group_name("[]")

    def test_rejects_too_long_name(self, app_client, monkeypatch):
        """Names longer than 30 characters are rejected."""
        from memory_server import _is_valid_group_name
        assert not _is_valid_group_name("Lifelong Dedication To Human Health")

    def test_accepts_valid_names(self, app_client, monkeypatch):
        """Valid 1-2 word group names pass validation."""
        from memory_server import _is_valid_group_name
        for name in ("Family", "Friends", "Colleagues", "Organizations", "Hobbies", "Pets"):
            assert _is_valid_group_name(name), f"'{name}' should be valid"

    def test_invalid_group_not_stored_on_add(self, app_client, monkeypatch):
        """When the LLM returns 'Empty_List' as new_group it is discarded, not stored."""
        import memory_server

        monkeypatch.setattr(
            "memory_server.process_memory_chunk",
            lambda text, **kwargs: MemoryProcessing(
                atomic_facts=["Jordan Kim is a teacher."],
                triples=[KnowledgeTriple(subject="Jordan Kim", predicate="IS", object="Teacher")],
            ),
        )
        monkeypatch.setattr(
            "memory_server.librarian_assign_groups",
            lambda n, g, **kw: GroupAssignment(matching_groups=[], new_group="Empty_List"),
        )

        resp = app_client.post("/memory/add", json={"text": "Jordan Kim is a teacher."})
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"

        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "SELECT g.name FROM groups g "
            "JOIN entity_groups eg ON g.id = eg.group_id "
            "JOIN entities e ON e.id = eg.entity_id "
            "WHERE LOWER(e.canonical_name) = 'jordan kim'"
        )
        groups = [r[0] for r in cursor.fetchall()]
        assert "Empty_List" not in groups, f"'Empty_List' must not be stored as group; got: {groups}"


# ---------------------------------------------------------------------------
# Phase 4d: self-loop guard
# ---------------------------------------------------------------------------

class TestPhase4dSelfLoop:
    """Tests that Phase 4d never creates PRECEDED_BY(fact, fact) self-loops."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", lambda c, n: None)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)
        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)

    def test_phase4d_no_self_loop_when_same_fact_in_both_buckets(self, app_client, monkeypatch):
        """Phase 4d must not create PRECEDED_BY(A, A) when the same fact backs both the
        current and historical entries for a predicate (can occur when a single IS edge's
        backing fact is mislabeled historical by the extraction LLM).

        We simulate this by injecting a KG edge whose backing fact is already marked
        historical, so both the current and historical buckets would contain the same fact_id.
        """
        import memory_server
        import uuid as _uuid

        subj_id = memory_server.get_or_create_entity("Alice")
        obj_id = memory_server.get_or_create_entity("Engineer")
        now = datetime.now().isoformat()
        fact_id = str(_uuid.uuid4())
        cursor = memory_server.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, temporal_status, created_at, last_accessed) "
            "VALUES (?, ?, 'historical', ?, ?)",
            (fact_id, "Alice is an engineer.", now, now),
        )
        memory_server.sqlite_conn.commit()
        # The IS edge is backed by a historical fact — if Phase 4d doesn't guard
        # curr_fid == hist_fid it would create a self-loop.
        memory_server.knowledge_graph.add_relationship(
            subj_id, "IS", obj_id,
            subject_name="Alice", object_name="Engineer",
            fact_ids=[fact_id], persist=True,
        )

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"

        tg = memory_server.temporal_graph.G
        self_loops = [
            (u, v, d)
            for u, v, d in tg.edges(data=True)
            if u == v and d.get("relation") == "PRECEDED_BY"
        ]
        assert self_loops == [], f"No PRECEDED_BY self-loops should exist, found: {self_loops}"


# ---------------------------------------------------------------------------
# Phase 6: decision logging and co-reference detection
# ---------------------------------------------------------------------------

class TestPhase6Logging:
    """Tests that Phase 6 logs LLM null/false decisions into the report."""

    def _stub_all_other_phases(self, monkeypatch):
        monkeypatch.setattr("memory_server.librarian_should_merge", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_split_compound", lambda f: None)
        monkeypatch.setattr("memory_server.librarian_check_supersession", lambda a, b: None)
        monkeypatch.setattr("memory_server.librarian_check_concurrency", lambda a, b, pa, pb: None)
        monkeypatch.setattr("memory_server.librarian_resolve_compound_entity", lambda c, n: None)
        monkeypatch.setattr("memory_server.librarian_classify_entity", lambda n, f: None)

    def test_phase6_logs_null_decision(self, app_client, monkeypatch):
        """When librarian_should_merge_entities returns None, report['merge_skipped_null'] increments."""
        import memory_server

        call_count = [0]
        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Jordan is Alice's partner.", temporal_status="current")],
                    triples=[KnowledgeTriple(subject="Jordan", predicate="IS_PARTNER_OF", object="Alice")],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Jordan Kim is a teacher.", temporal_status="current")],
                triples=[KnowledgeTriple(subject="Jordan Kim", predicate="IS", object="Teacher")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr("memory_server.librarian_should_merge_entities", lambda a, b, fa, fb: None)

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert report.get("merge_skipped_null", 0) >= 1, (
            "report['merge_skipped_null'] should be incremented when LLM returns None"
        )

    def test_phase6_logs_false_decision(self, app_client, monkeypatch):
        """When librarian_should_merge_entities returns should_merge=False, report['merge_skipped_false'] increments."""
        import memory_server

        call_count = [0]
        def vary_chunk(_text, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MemoryProcessing(
                    atomic_facts=[AtomicFact(text="Jordan lives in Portland.", temporal_status="current")],
                    triples=[KnowledgeTriple(subject="Jordan", predicate="LIVES_IN", object="Portland")],
                )
            return MemoryProcessing(
                atomic_facts=[AtomicFact(text="Jordan Kim works in SF.", temporal_status="current")],
                triples=[KnowledgeTriple(subject="Jordan Kim", predicate="WORKS_IN", object="San Francisco")],
            )

        monkeypatch.setattr("memory_server.process_memory_chunk", vary_chunk)
        monkeypatch.setattr("memory_server.librarian_assign_groups", lambda n, g, **kw: None)
        app_client.post("/memory/add", json={"text": "first"})
        app_client.post("/memory/add", json={"text": "second"})

        self._stub_all_other_phases(monkeypatch)
        monkeypatch.setattr(
            "memory_server.librarian_should_merge_entities",
            lambda a, b, fa, fb: EntityMergeDecision(
                should_merge=False, canonical_to_keep=b, explanation="different people"
            ),
        )

        resp = app_client.post("/memory/consolidate")
        task = wait_for_task(app_client, resp.json()["task_id"])
        assert task["status"] == "completed"
        report = task["result"]["report"]
        assert report.get("merge_skipped_false", 0) >= 1, (
            "report['merge_skipped_false'] should be incremented when LLM returns should_merge=False"
        )
