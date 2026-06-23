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
    ConcurrencyDecision,
    EntityExtraction,
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
            lambda name, groups: GroupAssignment(matching_groups=["Family"], new_group=None),
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
            lambda name, groups: GroupAssignment(matching_groups=[], new_group="Pets"),
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

        def counting_assign(name, groups):
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
            lambda name, groups: GroupAssignment(matching_groups=[], new_group="Friends"),
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
            lambda name, groups: GroupAssignment(matching_groups=[], new_group="Colleagues"),
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
            "memory_server.librarian_assign_groups", lambda name, groups: None
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
