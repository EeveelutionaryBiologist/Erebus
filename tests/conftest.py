"""
Shared fixtures for the Erebus test suite.

Marker quick-reference
----------------------
  @pytest.mark.requires_model  — needs GGUF files on disk
  @pytest.mark.integration     — hits real dbs (SQLite, ChromaDB, KG)
  @pytest.mark.slow            — expected runtime > a few seconds

Running subsets
---------------
  pytest                            # all tests (model tests will be collected but may skip)
  pytest -m "not requires_model"    # skip anything that needs GGUF files
  pytest tests/test_knowledge_graph.py  # graph unit tests only
"""

import chromadb
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from knowledge_graph import KnowledgeRelationshipGraph


# ---------------------------------------------------------------------------
# Knowledge-graph fixtures  (no server, no models)
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_path(tmp_path):
    """Path to a non-existent JSON file — KnowledgeRelationshipGraph creates it on first write."""
    return str(tmp_path / "graph.json")


@pytest.fixture
def fresh_graph(graph_path):
    """Empty KnowledgeRelationshipGraph backed by a tmp file."""
    return KnowledgeRelationshipGraph(graph_path)


# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated tmp dbs and all model calls stubbed.

    What is patched
    ---------------
    Storage
      SQLITE_PATH, CHROMA_PATH, GRAPH_DIR  → sub-dirs of tmp_path
      sqlite_conn   → fresh schema-initialised connection to tmp SQLite
      chroma_client / collection → fresh PersistentClient + collection in tmp
      knowledge_graph → empty KnowledgeRelationshipGraph on tmp file

    Models (nothing is downloaded or loaded)
      Llama            → MagicMock (prevents GGUF loading in startup_event)
      hf_hub_download  → MagicMock
      load_librarian_model → MagicMock
      get_embedding    → lambda returning a stable [0.1] * 768 vector

    Tests that need specific Librarian return values should further patch
    the relevant function (process_memory_chunk, extract_entities_from_text,
    librarian_should_merge, librarian_split_compound) via monkeypatch or
    the mocker fixture.
    """
    import memory_server

    db_dir = tmp_path / "DB"
    (db_dir / "chromadb").mkdir(parents=True)
    kg_dir = tmp_path / "KG"
    kg_dir.mkdir()

    # --- Redirect storage paths (must happen before init_sqlite / KG instantiation) ---
    monkeypatch.setattr(memory_server, "SQLITE_PATH", db_dir / "metadata.db")
    monkeypatch.setattr(memory_server, "CHROMA_PATH", db_dir / "chromadb")
    monkeypatch.setattr(memory_server, "GRAPH_DIR", kg_dir)

    # Fresh SQLite with the canonical schema
    test_conn = memory_server.init_sqlite()
    monkeypatch.setattr(memory_server, "sqlite_conn", test_conn)

    # Fresh ChromaDB
    test_chroma = chromadb.PersistentClient(path=str(db_dir / "chromadb"))
    test_collection = test_chroma.get_or_create_collection("nyxx_memory")
    monkeypatch.setattr(memory_server, "chroma_client", test_chroma)
    monkeypatch.setattr(memory_server, "collection", test_collection)

    # Fresh graph
    test_graph = KnowledgeRelationshipGraph(str(kg_dir / "knowledge_graph.json"))
    monkeypatch.setattr(memory_server, "knowledge_graph", test_graph)

    # Stub embedding (deterministic; 768-dim matches bge-base-en-v1.5)
    monkeypatch.setattr(memory_server, "get_embedding", lambda _: [0.1] * 768)

    # Stub model loading so startup_event never tries to read GGUF files
    monkeypatch.setattr(memory_server, "Llama", MagicMock)
    monkeypatch.setattr(memory_server, "hf_hub_download", MagicMock())
    monkeypatch.setattr(memory_server, "load_llm_client", MagicMock())

    # Default: no context hint (tests that need hint behaviour override this)
    monkeypatch.setattr(memory_server, "extract_context_hint", lambda text: None)

    # Run background tasks synchronously so side effects are committed before
    # the HTTP response arrives. Tests that check task results use wait_for_task().
    monkeypatch.setattr(
        memory_server,
        "_run_task_in_background",
        lambda task_id, fn, *args, **kwargs: memory_server._execute_task(task_id, fn, *args, **kwargs),
    )

    with TestClient(memory_server.app) as client:
        yield client
