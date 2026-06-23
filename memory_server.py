import re
import json
import uuid
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import networkx as nx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from llm_client import load_llm_client
from librarian import (
    process_memory_chunk,
    extract_entities_from_text,
    extract_context_hint,
    librarian_summarize,
    librarian_should_merge,
    librarian_split_compound,
    librarian_check_supersession,
    librarian_assign_groups,
    ContextHint,
)
from knowledge_graph import KnowledgeRelationshipGraph

CONSOLIDATION_PASSES = 2

# ==========================================
# 1. DIRECTORY SETUP & CONFIGURATION
# ==========================================
app = FastAPI(title="Erebus Memory Microservice")

BASE_DIR = Path(__file__).resolve().parent

MEMORY_DIR = BASE_DIR / "DB"
GRAPH_DIR = BASE_DIR / "KnowledgeGraph"
CHROMA_PATH = MEMORY_DIR / "chromadb"
SQLITE_PATH = MEMORY_DIR / "metadata.db"
EMBEDDING_DIR = BASE_DIR / "Embedding"

GGUF_MODEL_PATH = EMBEDDING_DIR / "bge-base-en-v1.5-f16.gguf"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. INITIALIZE MODELS & GRAPH
# ==========================================
embedder = None

@app.on_event("startup")
def startup_event():
    """Runs on Uvicorn startup."""
    global embedder
    
    # Load Embedding Model
    if not GGUF_MODEL_PATH.exists():
        print("[SYSTEM] Downloading embedding model...")
        hf_hub_download(
            repo_id="CompendiumLabs/bge-base-en-v1.5-gguf",
            filename="bge-base-en-v1.5-f16.gguf",
            local_dir=EMBEDDING_DIR
        )
    print("[SYSTEM] Initializing Llama.cpp Embedder...")
    embedder = Llama(model_path=str(GGUF_MODEL_PATH), embedding=True, verbose=False)
    
    # Load LLM backend (local Qwen or cloud provider per config.json)
    load_llm_client()
    _migrate_to_v2()
    _migrate_to_v3()
    _migrate_to_v4()

def get_embedding(text: str) -> list[float]:
    response = embedder.create_embedding(text)
    return response["data"][0]["embedding"]

# Initialize Knowledge Graph
knowledge_graph = KnowledgeRelationshipGraph(str(GRAPH_DIR / "knowledge_graph.json"))
# Temporal graph: state-instance nodes (fact_ids), PRECEDED_BY edges for supersession history
temporal_graph = KnowledgeRelationshipGraph(str(GRAPH_DIR / "temporal_graph.json"))

# ==========================================
# 3. DATABASE INITIALIZATION
# ==========================================
def init_sqlite():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS raw_chunks (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            created_at  DATETIME,
            last_accessed DATETIME
        );
        CREATE TABLE IF NOT EXISTS atomic_facts (
            id              TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            hit_count       INTEGER DEFAULT 0,
            temporal_status TEXT DEFAULT 'current',
            valid_period    TEXT,
            source_chunk_id TEXT REFERENCES raw_chunks(id),
            created_at      DATETIME,
            last_accessed   DATETIME
        );
        CREATE TABLE IF NOT EXISTS entities (
            id             TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            aliases        TEXT DEFAULT '[]',
            hit_count      INTEGER DEFAULT 0,
            created_at     DATETIME,
            last_accessed  DATETIME
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_ci
            ON entities(LOWER(canonical_name));
        CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_id  TEXT NOT NULL REFERENCES entities(id),
            chunk_id   TEXT NOT NULL REFERENCES raw_chunks(id),
            PRIMARY KEY (entity_id, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS groups (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            created_at DATETIME
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_groups_name_ci
            ON groups(LOWER(name));
        CREATE TABLE IF NOT EXISTS entity_groups (
            entity_id  TEXT NOT NULL REFERENCES entities(id),
            group_id   TEXT NOT NULL REFERENCES groups(id),
            PRIMARY KEY (entity_id, group_id)
        );
    """)
    conn.commit()
    return conn

sqlite_conn = init_sqlite()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="nyxx_memory")

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
)

def _is_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(s.lower()))

def _migrate_kg_nodes(cursor):
    """Creates entity rows for legacy string-keyed KG nodes and rebuilds the graph with UUID keys."""
    now = datetime.now().isoformat()
    name_to_id: dict[str, str] = {}

    for raw_name in list(knowledge_graph.G.nodes()):
        name_str = str(raw_name)
        # Use get-or-create so case-variants collapse to one entity.
        cursor.execute("SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)", (name_str,))
        row = cursor.fetchone()
        if row:
            name_to_id[name_str] = row[0]
        else:
            entity_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
                "VALUES (?, ?, '[]', 0, ?, ?)",
                (entity_id, name_str, now, now)
            )
            name_to_id[name_str] = entity_id

    knowledge_graph.rebuild_with_name_to_id_mapping(name_to_id)
    print(f"[SYSTEM] KG migrated: {len(name_to_id)} nodes converted to UUID keys.")

def _migrate_to_v2():
    """Migrates the legacy unified 'memories' table to the three-table v2 schema."""
    cursor = sqlite_conn.cursor()

    # Migrate legacy data if the old table still exists.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
    if cursor.fetchone():
        print("[SYSTEM] Migrating legacy 'memories' table to v2 schema...")
        cursor.execute("""
            INSERT OR IGNORE INTO raw_chunks (id, content, created_at, last_accessed)
            SELECT id, content, created_at, last_accessed FROM memories WHERE record_type = 'raw'
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO atomic_facts (id, content, hit_count, created_at, last_accessed)
            SELECT id, content, hit_count, created_at, last_accessed FROM memories WHERE record_type = 'fact'
        """)
        cursor.execute("DROP TABLE memories")
        sqlite_conn.commit()
        print("[SYSTEM] Legacy 'memories' table migrated and dropped.")

    # Migrate KG nodes if any are still string-keyed (not UUIDs).
    nodes = list(knowledge_graph.G.nodes())
    if nodes and not all(_is_uuid(str(n)) for n in nodes):
        _migrate_kg_nodes(cursor)
        sqlite_conn.commit()

    # Add temporal columns to atomic_facts if absent (idempotent — safe to run every startup).
    cursor.execute("PRAGMA table_info(atomic_facts)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "temporal_status" not in existing_cols:
        cursor.execute("ALTER TABLE atomic_facts ADD COLUMN temporal_status TEXT DEFAULT 'current'")
    if "valid_period" not in existing_cols:
        cursor.execute("ALTER TABLE atomic_facts ADD COLUMN valid_period TEXT")
    sqlite_conn.commit()

def _migrate_to_v3():
    """Adds source_chunk_id to atomic_facts and creates the entity_chunks join table."""
    cursor = sqlite_conn.cursor()
    cursor.execute("PRAGMA table_info(atomic_facts)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "source_chunk_id" not in existing_cols:
        cursor.execute(
            "ALTER TABLE atomic_facts ADD COLUMN source_chunk_id TEXT REFERENCES raw_chunks(id)"
        )
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_id  TEXT NOT NULL REFERENCES entities(id),
            chunk_id   TEXT NOT NULL REFERENCES raw_chunks(id),
            PRIMARY KEY (entity_id, chunk_id)
        )
    """)
    sqlite_conn.commit()

def _migrate_to_v4():
    """Creates the groups and entity_groups tables for thematic entity clustering."""
    cursor = sqlite_conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS groups (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            created_at DATETIME
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_groups_name_ci
            ON groups(LOWER(name));
        CREATE TABLE IF NOT EXISTS entity_groups (
            entity_id  TEXT NOT NULL REFERENCES entities(id),
            group_id   TEXT NOT NULL REFERENCES groups(id),
            PRIMARY KEY (entity_id, group_id)
        );
    """)
    sqlite_conn.commit()

# ---------------------------------------------------------------------------
# Write-time normalization
# ---------------------------------------------------------------------------

# Syntactic-variant collapse only — same-direction, no semantic rewrites.
_PREDICATE_SYNONYMS: dict[str, str] = {
    "HAVE":      "HAS",
    "HAVE_A":    "HAS",
    "HAS_A":     "HAS",
    "IS_A":      "IS",
    "IS_AN":     "IS",
    "WORKS_FOR": "WORKS_AT",
}

def normalize_entity_name(name: str) -> str:
    """Title-cases an entity name so 'hailey' and 'HAILEY' both become 'Hailey'."""
    return name.strip().title()

def normalize_predicate(pred: str) -> str:
    """Uppercases a predicate and collapses known syntactic variants to a canonical form."""
    normalized = re.sub(r"\s+", "_", pred.strip().upper())
    return _PREDICATE_SYNONYMS.get(normalized, normalized)

# ---------------------------------------------------------------------------

def get_or_create_entity(name: str, conn: sqlite3.Connection | None = None) -> str:
    """Returns the entity UUID for `name`, inserting a new row if it doesn't exist."""
    c = (conn or sqlite_conn)
    cursor = c.cursor()
    cursor.execute("SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    entity_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO entities (id, canonical_name, aliases, hit_count, created_at, last_accessed) "
        "VALUES (?, ?, '[]', 0, ?, ?)",
        (entity_id, name, now, now)
    )
    c.commit()
    return entity_id

def get_or_create_group(name: str, conn: sqlite3.Connection | None = None) -> str:
    """Returns the group UUID for `name` (case-insensitive), inserting a new row if absent."""
    c = conn or sqlite_conn
    cursor = c.cursor()
    normalized = name.strip().title()
    cursor.execute("SELECT id FROM groups WHERE LOWER(name) = LOWER(?)", (normalized,))
    row = cursor.fetchone()
    if row:
        return row[0]
    group_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO groups (id, name, created_at) VALUES (?, ?, ?)",
        (group_id, normalized, datetime.now().isoformat()),
    )
    c.commit()
    return group_id

def lookup_entity(name: str) -> str | None:
    """Returns the entity UUID for `name` (case-insensitive), or None if not found."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id FROM entities WHERE LOWER(canonical_name) = LOWER(?)", (name,))
    row = cursor.fetchone()
    return row[0] if row else None

# ==========================================
# 4. CONSOLIDATION CONFIGURATION
# ==========================================
# Facts older than this with zero hits are pruned.
PRUNE_AGE_DAYS = 60

# Cosine similarity threshold above which two facts are sent to the Librarian
# for a merge decision. Range: 0.0–1.0. Higher = more conservative.
DEDUP_SIMILARITY_THRESHOLD = 0.90

# Above this threshold the facts are considered near-identical text; the Librarian
# is skipped and the lower-hit-count copy is dropped directly.
HIGH_SIM_DEDUP_THRESHOLD = 0.99

# Only facts longer than this are checked for compound structure (short facts are
# almost always already atomic, so skip the Librarian call to save CPU).
COMPOUND_CHECK_MIN_CHARS = 120

# /memory/learn chunking
LEARN_CHUNK_SIZE = 5  # target sentences per chunk (no overlap to avoid duplicate ingestion)

# Phase 4 (Supersession / Contradiction)
# Maps a past-tense predicate to its present-tense counterpart. When both exist for the
# same (subject, object) entity pair, the past-tense edge's source facts are marked historical.
TEMPORAL_PREDICATE_PAIRS: dict[str, str] = {
    "WAS": "IS",
    "HAD": "HAS",
}

# Predicate pairs that represent direct logical contradiction (no temporal ordering implied).
# Both source fact sets are added to report["flagged"] for human review.
CONTRADICTION_PREDICATE_PAIRS: list[tuple[str, str]] = [
    ("IS", "IS_NOT"),
    ("HAS", "HAS_NOT"),
]

# Words that signal a fact describes a state that has changed. Facts containing these
# phrases are candidates for text-based supersession checks in Phase 4b.
SUPERSESSION_KEYWORDS: frozenset[str] = frozenset([
    "no longer", "used to", "formerly", "previously", "once was", "not anymore",
])

# ==========================================
# 5. ASYNC TASK INFRASTRUCTURE
# ==========================================

_task_registry: dict[str, dict[str, Any]] = {}
_task_lock = threading.Lock()


def _create_task() -> str:
    task_id = str(uuid.uuid4())
    with _task_lock:
        _task_registry[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
        }
    return task_id


def _execute_task(task_id: str, fn, *args, **kwargs):
    """Run fn(*args, conn=conn, **kwargs) in the caller's thread, updating task registry."""
    with _task_lock:
        _task_registry[task_id]["status"] = "running"
    conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
    try:
        result = fn(*args, conn=conn, **kwargs)
        with _task_lock:
            _task_registry[task_id].update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat(),
            })
    except HTTPException as e:
        with _task_lock:
            _task_registry[task_id].update({
                "status": "failed",
                "error": e.detail,
                "completed_at": datetime.now().isoformat(),
            })
    except Exception as e:
        with _task_lock:
            _task_registry[task_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            })
    finally:
        conn.close()


def _run_task_in_background(task_id: str, fn, *args, **kwargs):
    """Dispatch fn to a daemon thread; call _execute_task (monkeypatch this in tests)."""
    threading.Thread(
        target=_execute_task,
        args=(task_id, fn) + args,
        kwargs=kwargs,
        daemon=True,
    ).start()


@app.get("/memory/task/{task_id}")
def get_task_status(task_id: str):
    """Poll the status of a background task created by /add, /learn, or /consolidate."""
    with _task_lock:
        task = dict(_task_registry.get(task_id, {}))
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")
    return task


# ==========================================
# 6. API ENDPOINTS
# ==========================================
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Splits text on sentence boundaries into chunks of up to chunk_size sentences."""
    sentences = [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def _format_context_hint(hint: ContextHint) -> str | None:
    """Formats a ContextHint as a '[CONTEXT: ...]' prefix, or None if both fields are empty."""
    parts = [p for p in (hint.subject, hint.time_period) if p]
    return f"[CONTEXT: {', '.join(parts)}] " if parts else None

class MemoryInput(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

def _add_memory_sync(memory: MemoryInput, conn: sqlite3.Connection) -> dict:
    """Synchronous core of /memory/add. Called by the background task runner."""
    now = datetime.now().isoformat()
    cursor = conn.cursor()

    # 1. Ask Librarian to process the chunk
    processed_data = process_memory_chunk(memory.text)
    if not processed_data:
        raise HTTPException(status_code=500, detail="Librarian failed to process memory.")

    # 2. Store original raw chunk in SQLite (provenance record, not indexed in ChromaDB)
    raw_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO raw_chunks (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
        (raw_id, memory.text, now, now)
    )
    conn.commit()

    # 3. Save Atomic Facts to ChromaDB & SQLite
    fact_ids_batch = []
    for fact in processed_data.atomic_facts:
        fact_id = str(uuid.uuid4())
        fact_ids_batch.append(fact_id)
        vector = get_embedding(fact.text)

        collection.add(embeddings=[vector], documents=[fact.text], ids=[fact_id])
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, source_chunk_id, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fact_id, fact.text, fact.temporal_status, fact.valid_period, raw_id, now, now)
        )
    conn.commit()

    # 4. Save Triples to Knowledge Graph, linked to this batch's fact IDs.
    # persist=False defers the disk write; we flush once after the loop.
    entity_ids_in_chunk: dict[str, str] = {}  # entity_id → canonical_name
    for triple in processed_data.triples:
        subj = normalize_entity_name(triple.subject)
        obj  = normalize_entity_name(triple.object)
        pred = normalize_predicate(triple.predicate)
        subject_id = get_or_create_entity(subj, conn=conn)
        object_id  = get_or_create_entity(obj, conn=conn)
        entity_ids_in_chunk[subject_id] = subj
        entity_ids_in_chunk[object_id] = obj
        cursor.execute(
            "INSERT OR IGNORE INTO entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
            (subject_id, raw_id),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO entity_chunks (entity_id, chunk_id) VALUES (?, ?)",
            (object_id, raw_id),
        )
        knowledge_graph.add_relationship(
            subject_id, pred, object_id,
            subject_name=subj, object_name=obj,
            fact_ids=fact_ids_batch,
            persist=False,
        )
        print(f"  -> Graph Mapped: {subj} [{pred}] {obj}")
    if processed_data.triples:
        conn.commit()
        knowledge_graph.write_graph()

    # 5. Assign thematic groups to new entities (entities not yet in entity_groups).
    if entity_ids_in_chunk:
        cursor.execute("SELECT name FROM groups ORDER BY name")
        existing_groups = [r[0] for r in cursor.fetchall()]
        for entity_id, entity_name in entity_ids_in_chunk.items():
            cursor.execute(
                "SELECT COUNT(*) FROM entity_groups WHERE entity_id = ?", (entity_id,)
            )
            if cursor.fetchone()[0] > 0:
                continue
            assignment = librarian_assign_groups(entity_name, existing_groups)
            if not assignment:
                continue
            for group_name in assignment.matching_groups:
                group_id = get_or_create_group(group_name, conn=conn)
                cursor.execute(
                    "INSERT OR IGNORE INTO entity_groups (entity_id, group_id) VALUES (?, ?)",
                    (entity_id, group_id),
                )
            if assignment.new_group:
                group_id = get_or_create_group(assignment.new_group, conn=conn)
                cursor.execute(
                    "INSERT OR IGNORE INTO entity_groups (entity_id, group_id) VALUES (?, ?)",
                    (entity_id, group_id),
                )
        conn.commit()

    return {
        "status": "success",
        "message": f"Added {len(processed_data.atomic_facts)} standalone facts and {len(processed_data.triples)} graph relations.",
        "facts_added": len(processed_data.atomic_facts),
        "triples_added": len(processed_data.triples),
    }


@app.post("/memory/add", status_code=202)
def add_memory(memory: MemoryInput):
    """Enqueues text for Librarian processing. Returns a task handle immediately.

    Poll GET /memory/task/{task_id} for status and results.
    """
    task_id = _create_task()
    _run_task_in_background(task_id, _add_memory_sync, memory)
    return {"task_id": task_id, "status": "pending"}


def _learn_from_source_sync(memory: MemoryInput, conn: sqlite3.Connection) -> dict:
    """Synchronous core of /memory/learn. Called by the background task runner."""
    chunks = _split_into_chunks(memory.text, LEARN_CHUNK_SIZE)
    if not chunks:
        return {"status": "success", "chunks_total": 0, "chunks_succeeded": 0,
                "facts_added": 0, "triples_added": 0, "errors": []}

    context_prefix: str | None = None
    if len(chunks) > 1:
        hint = extract_context_hint(chunks[0])
        if hint:
            context_prefix = _format_context_hint(hint)

    facts_added = 0
    triples_added = 0
    errors: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_text = (context_prefix + chunk) if (context_prefix and i > 0) else chunk
        try:
            result = _add_memory_sync(MemoryInput(text=chunk_text), conn=conn)
            facts_added += result["facts_added"]
            triples_added += result["triples_added"]
        except HTTPException as e:
            errors.append({"chunk_index": i, "text": chunk[:80], "error": e.detail})

    return {
        "status": "success" if not errors else "partial",
        "chunks_total": len(chunks),
        "chunks_succeeded": len(chunks) - len(errors),
        "facts_added": facts_added,
        "triples_added": triples_added,
        "errors": errors,
    }


@app.post("/memory/learn", status_code=202)
def learn_from_source(memory: MemoryInput):
    """Splits large text into sentence-boundary chunks and enqueues them for processing.

    For multi-chunk inputs, a [CONTEXT: subject, time_period] prefix extracted from chunk 0
    is prepended to chunks 1+ to ground pronoun resolution across chunk boundaries.
    Poll GET /memory/task/{task_id} for status and results.
    """
    task_id = _create_task()
    _run_task_in_background(task_id, _learn_from_source_sync, memory)
    return {"task_id": task_id, "status": "pending"}

@app.post("/memory/search")
def search_memory(search: SearchQuery):
    """Searches ChromaDB (vectors) and Knowledge Graph (relations)."""
    now = datetime.now().isoformat()
    cursor = sqlite_conn.cursor()
    
    # --- 1. VECTOR SEARCH ---
    query_vector = get_embedding(search.query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=search.top_k
    )
    
    final_results = []
    if results['ids'] and results['ids'][0]:
        for mem_id in results['ids'][0]:
            cursor.execute(
                "UPDATE atomic_facts SET hit_count = hit_count + 1, last_accessed = ? WHERE id = ?",
                (now, mem_id)
            )
            cursor.execute(
                "SELECT content, hit_count, source_chunk_id FROM atomic_facts WHERE id = ?", (mem_id,)
            )
            row = cursor.fetchone()
            if row:
                final_results.append({
                    "id": mem_id,
                    "text": row[0], "hit_count": row[1], "source_chunk_id": row[2],
                })
    sqlite_conn.commit()

    # --- 2. GRAPH RETRIEVAL ---
    relation_facts = []
    entity_groups_found: dict[str, list[str]] = {}  # entity_name → [group_name, ...]
    extracted = extract_entities_from_text(search.query)

    if extracted and hasattr(extracted, 'entities'):
        for entity in extracted.entities:
            entity_id = lookup_entity(entity.name)
            if not entity_id:
                continue
            facts = knowledge_graph.retrieve_relationships(entity_id, depth=1)
            if facts:
                relation_facts.extend(facts)
                cursor.execute(
                    "UPDATE entities SET hit_count = hit_count + 1, last_accessed = ? WHERE id = ?",
                    (now, entity_id)
                )
            cursor.execute("""
                SELECT g.name FROM entity_groups eg
                JOIN groups g ON eg.group_id = g.id
                WHERE eg.entity_id = ?
                ORDER BY g.name
            """, (entity_id,))
            groups = [r[0] for r in cursor.fetchall()]
            if groups:
                cursor.execute(
                    "SELECT canonical_name FROM entities WHERE id = ?", (entity_id,)
                )
                name_row = cursor.fetchone()
                if name_row:
                    entity_groups_found[name_row[0]] = groups
    sqlite_conn.commit()

    summarized_context = ""
    if relation_facts:
        unique_facts = list(set(relation_facts))
        summarized_context = "\n".join(unique_facts)

    # --- 3. TEMPORAL CONTEXT ---
    # For each returned fact that appears in the temporal graph as a current-state node
    # (i.e., has outgoing PRECEDED_BY edges), surface its historical predecessors.
    # Predecessor texts are fetched live from SQLite; dead endpoints (fact deleted by a
    # later consolidation phase) are silently skipped.
    temporal_context: list[dict] = []
    for result in final_results:
        fact_id = result["id"]
        if not temporal_graph.G.has_node(fact_id):
            continue
        ancestor_ids = nx.descendants(temporal_graph.G, fact_id)
        if not ancestor_ids:
            continue
        ancestor_texts: list[str] = []
        for anc_id in ancestor_ids:
            cursor.execute("SELECT content FROM atomic_facts WHERE id = ?", (anc_id,))
            anc_row = cursor.fetchone()
            if anc_row:
                ancestor_texts.append(anc_row[0])
        if ancestor_texts:
            temporal_context.append({
                "current_fact": result["text"],
                "preceded_by": ancestor_texts,
            })

    return {
        "results": final_results,
        "relational_context": summarized_context,
        "entity_groups": entity_groups_found,
        "temporal_context": temporal_context,
    }

def _extract_entity_candidates(query: str) -> list[str]:
    """Tokenizes a query into words and bigrams for entity lookup without the Librarian."""
    words = re.findall(r"[a-zA-Z']+", query)
    candidates = list(words)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i + 1]}")
    # Deduplicate case-insensitively while preserving first-occurrence order.
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result

@app.post("/memory/context")
def context_memory(search: SearchQuery):
    """Fast-path retrieval for Nyxx: vector search + regex-driven depth-1 graph lookup.

    No Librarian calls. Designed to stay well under 100ms on cached embeddings.
    Hit counts are updated so pruning reflects actual usage across both retrieval paths.
    """
    now = datetime.now().isoformat()
    cursor = sqlite_conn.cursor()

    # --- 1. VECTOR SEARCH ---
    query_vector = get_embedding(search.query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=search.top_k
    )

    final_results = []
    if results["ids"] and results["ids"][0]:
        for mem_id in results["ids"][0]:
            # Only surface current facts; historical facts remain in ChromaDB for temporal queries
            # but are excluded from the fast-path context window.
            cursor.execute(
                "UPDATE atomic_facts SET hit_count = hit_count + 1, last_accessed = ? "
                "WHERE id = ? AND temporal_status = 'current'",
                (now, mem_id),
            )
            cursor.execute(
                "SELECT content, hit_count FROM atomic_facts WHERE id = ? AND temporal_status = 'current'",
                (mem_id,),
            )
            row = cursor.fetchone()
            if row:
                final_results.append({"text": row[0], "hit_count": row[1]})

    # --- 2. GRAPH RETRIEVAL (no Librarian — regex candidates only) ---
    relation_facts: list[str] = []
    for candidate in _extract_entity_candidates(search.query):
        entity_id = lookup_entity(candidate)
        if not entity_id:
            continue
        facts = knowledge_graph.retrieve_relationships(entity_id, depth=1)
        if facts:
            relation_facts.extend(facts)
            cursor.execute(
                "UPDATE entities SET hit_count = hit_count + 1, last_accessed = ? WHERE id = ?",
                (now, entity_id),
            )

    sqlite_conn.commit()

    # Deduplicate while preserving encounter order.
    relational_context = "\n".join(dict.fromkeys(relation_facts))

    return {
        "results": final_results,
        "relational_context": relational_context,
    }

@app.get("/memory/all")
def get_all_memories(type: Literal["raw", "fact", "entity"] | None = None):
    """Retrieves records from SQLite. Optionally filtered by ?type=raw|fact|entity.

    Omitting ?type returns all three record types in a flat list (original behaviour).
    """
    cursor = sqlite_conn.cursor()
    results = []

    if type is None or type == "raw":
        cursor.execute("SELECT id, content, 0, created_at FROM raw_chunks ORDER BY created_at DESC")
        results += [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3], "record_type": "raw"} for r in cursor.fetchall()]

    if type is None or type == "fact":
        cursor.execute(
            "SELECT id, content, hit_count, created_at, source_chunk_id FROM atomic_facts ORDER BY created_at DESC"
        )
        results += [
            {"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3],
             "source_chunk_id": r[4], "record_type": "fact"}
            for r in cursor.fetchall()
        ]

    if type is None or type == "entity":
        cursor.execute("""
            SELECT e.id, e.canonical_name, e.hit_count, e.created_at, COUNT(DISTINCT ec.chunk_id)
            FROM entities e
            LEFT JOIN entity_chunks ec ON e.id = ec.entity_id
            GROUP BY e.id
            ORDER BY e.created_at DESC
        """)
        entity_rows = cursor.fetchall()
        for r in entity_rows:
            cursor.execute("""
                SELECT g.name FROM entity_groups eg
                JOIN groups g ON eg.group_id = g.id
                WHERE eg.entity_id = ?
                ORDER BY g.name
            """, (r[0],))
            groups = [row[0] for row in cursor.fetchall()]
            results.append({
                "id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3],
                "chunk_count": r[4], "groups": groups, "record_type": "entity",
            })

    return {"results": results}

@app.delete("/memory/clear")
def clear_all_memories():
    """Wipes ChromaDB, SQLite, and the Knowledge Graph completely."""
    # 1. Clear SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute("DELETE FROM raw_chunks")
    cursor.execute("DELETE FROM atomic_facts")
    cursor.execute("DELETE FROM entities")
    sqlite_conn.commit()
    
    # 2. Clear ChromaDB
    global collection
    chroma_client.delete_collection("nyxx_memory")
    collection = chroma_client.create_collection("nyxx_memory")
    
    # 3. Clear Graphs
    knowledge_graph.clear()
    temporal_graph.clear()

    return {"status": "success", "message": "All databases and graphs wiped clean."}


def _consolidate_memories_sync(conn: sqlite3.Connection) -> dict:
    """Synchronous core of /memory/consolidate. Called by the background task runner.

    Five-phase memory hygiene pass:
      Phase 0 – Exact dedup: collapse identical fact text to a single row.
      Phase 1 – Prune: delete stale, never-retrieved atomic facts from ChromaDB + SQLite.
      Phase 2 – Merge: detect near-duplicate fact pairs via cosine similarity; ask the
                       Librarian to produce a merged fact and replace the originals.
      Phase 3 – Split: find compound facts and break them into atomic sentences.
      Phase 4 – Supersession/Contradiction: scan KG edges for IS/WAS (and IS/IS_NOT) pairs
                       on the same entity pair. WAS source facts are marked 'historical';
                       opposing predicates are flagged for review.

    KG source tracking: each KG edge stores source_fact_ids. Deleting a fact calls
    knowledge_graph.remove_fact_reference(), which removes it from all edges and deletes
    edges whose source list becomes empty. Legacy edges (no source_fact_ids) are left
    in place and cleaned up by the degree=0 orphan sweep at the end of each pass.
    """
    report = {"pruned": 0, "merged": 0, "split": 0, "superseded": 0, "flagged": [], "errors": []}
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    # Tracks contradiction pairs already added to report["flagged"] so multi-pass doesn't duplicate.
    flagged_pairs: set[tuple[str, str, str, str]] = set()
    # Fact IDs involved in a detected contradiction — protected from Phase 2 dedup since they
    # are intentionally distinct facts, not duplicates.
    protected_fact_ids: set[str] = set()
    # Fact-ID pairs already sent to librarian_check_supersession for text-based checks.
    # Prevents re-checking the same pair across multiple consolidation passes.
    flagged_text_pairs: set[tuple[str, str]] = set()

    for _N in range(CONSOLIDATION_PASSES):
        # ------------------------------------------------------------------
        # Phase 4: Structural supersession and contradiction detection via KG
        # Runs first so Phase 2 cannot destroy IS/WAS or IS/IS_NOT pairs before we inspect them.
        # ------------------------------------------------------------------
        print("[CONSOLIDATE] Phase 4: Structural supersession/contradiction detection...")

        for subject_id in list(knowledge_graph.G.nodes()):
            for object_id in list(knowledge_graph.G.successors(subject_id)):
                if not knowledge_graph.G.has_edge(subject_id, object_id):
                    continue
                edges = knowledge_graph.G[subject_id][object_id]

                # Index edge data by predicate for fast membership tests.
                predicate_map: dict[str, list[dict]] = {}
                for _k, data in edges.items():
                    pred = data.get("relation", "")
                    predicate_map.setdefault(pred, []).append(data)

                subj_name = knowledge_graph.G.nodes[subject_id].get("name", subject_id)
                obj_name = knowledge_graph.G.nodes[object_id].get("name", object_id)

                # Supersession: past-tense predicate coexists with its present-tense counterpart.
                for past_pred, present_pred in TEMPORAL_PREDICATE_PAIRS.items():
                    if past_pred not in predicate_map or present_pred not in predicate_map:
                        continue
                    # Collect present-state fact IDs once per predicate pair for temporal linking.
                    current_fact_ids_for_pair: list[str] = []
                    for present_edge_data in predicate_map[present_pred]:
                        current_fact_ids_for_pair.extend(
                            present_edge_data.get("source_fact_ids", [])
                        )
                    for edge_data in predicate_map[past_pred]:
                        for fact_id in edge_data.get("source_fact_ids", []):
                            cursor.execute(
                                "SELECT temporal_status FROM atomic_facts WHERE id = ?", (fact_id,)
                            )
                            row = cursor.fetchone()
                            if row and row[0] != "historical":
                                cursor.execute(
                                    "UPDATE atomic_facts SET temporal_status = 'historical' WHERE id = ?",
                                    (fact_id,),
                                )
                                report["superseded"] += 1
                                # Temporal graph: current -[PRECEDED_BY]-> past
                                cursor.execute(
                                    "SELECT content FROM atomic_facts WHERE id = ?", (fact_id,)
                                )
                                past_row = cursor.fetchone()
                                past_name = past_row[0][:80] if past_row else fact_id[:8]
                                for current_fact_id in current_fact_ids_for_pair:
                                    if current_fact_id == fact_id:
                                        continue
                                    cursor.execute(
                                        "SELECT content FROM atomic_facts WHERE id = ?",
                                        (current_fact_id,),
                                    )
                                    cur_row = cursor.fetchone()
                                    cur_name = cur_row[0][:80] if cur_row else current_fact_id[:8]
                                    temporal_graph.add_relationship(
                                        current_fact_id, "PRECEDED_BY", fact_id,
                                        subject_name=cur_name, object_name=past_name,
                                        fact_ids=[fact_id], persist=False,
                                    )
                                print(
                                    f"[CONSOLIDATE] Superseded: {subj_name} [{past_pred}] {obj_name} "
                                    f"(fact {fact_id[:8]}…) overridden by [{present_pred}] edge."
                                )

                # Contradiction: opposing predicates — flag for human review, no auto-resolution.
                for pred_a, pred_b in CONTRADICTION_PREDICATE_PAIRS:
                    if pred_a not in predicate_map or pred_b not in predicate_map:
                        continue
                    pair_key = (subject_id, object_id, pred_a, pred_b)
                    if pair_key in flagged_pairs:
                        continue
                    flagged_pairs.add(pair_key)
                    facts_a = []
                    for edge_data in predicate_map[pred_a]:
                        for fact_id in edge_data.get("source_fact_ids", []):
                            protected_fact_ids.add(fact_id)
                            cursor.execute("SELECT content FROM atomic_facts WHERE id = ?", (fact_id,))
                            row = cursor.fetchone()
                            if row:
                                facts_a.append(row[0])
                    facts_b = []
                    for edge_data in predicate_map[pred_b]:
                        for fact_id in edge_data.get("source_fact_ids", []):
                            protected_fact_ids.add(fact_id)
                            cursor.execute("SELECT content FROM atomic_facts WHERE id = ?", (fact_id,))
                            row = cursor.fetchone()
                            if row:
                                facts_b.append(row[0])
                    if facts_a and facts_b:
                        report["flagged"].append({
                            "type": "contradiction",
                            "subject": subj_name,
                            "object": obj_name,
                            "predicate_a": pred_a,
                            "predicate_b": pred_b,
                            "facts_a": facts_a,
                            "facts_b": facts_b,
                        })
                        print(
                            f"[CONSOLIDATE] Contradiction: {subj_name} [{pred_a}] vs [{pred_b}] {obj_name}"
                        )

        conn.commit()

        # ------------------------------------------------------------------
        # Phase 4b: Text-based supersession via librarian_check_supersession
        # Handles same-predicate or KG-absent cases where the fact text itself
        # signals state change ("no longer", "used to", etc.).
        # ------------------------------------------------------------------
        print("[CONSOLIDATE] Phase 4b: Text-based supersession detection...")

        cursor.execute(
            "SELECT id, content FROM atomic_facts WHERE temporal_status != 'historical'"
        )
        current_facts = cursor.fetchall()

        keyword_facts = [
            (fid, text) for fid, text in current_facts
            if any(kw in text.lower() for kw in SUPERSESSION_KEYWORDS)
            and fid not in protected_fact_ids
        ]

        for fact_id, fact_text in keyword_facts:
            # Re-check: a prior iteration may have already marked this fact historical.
            cursor.execute(
                "SELECT temporal_status FROM atomic_facts WHERE id = ?", (fact_id,)
            )
            row = cursor.fetchone()
            if not row or row[0] == "historical":
                continue

            n_available = collection.count()
            if n_available < 2:
                continue
            try:
                query_result = collection.query(
                    query_embeddings=[get_embedding(fact_text)],
                    n_results=min(5, n_available),
                )
            except Exception:
                continue

            neighbor_ids = query_result["ids"][0] if query_result["ids"] else []
            neighbor_docs = query_result["documents"][0] if query_result["documents"] else []

            for neighbor_id, neighbor_text in zip(neighbor_ids, neighbor_docs):
                if neighbor_id == fact_id:
                    continue
                if neighbor_id in protected_fact_ids:
                    continue

                pair_key = tuple(sorted((fact_id, neighbor_id)))
                if pair_key in flagged_text_pairs:
                    continue
                flagged_text_pairs.add(pair_key)

                cursor.execute(
                    "SELECT temporal_status FROM atomic_facts WHERE id = ?", (neighbor_id,)
                )
                row = cursor.fetchone()
                if not row or row[0] == "historical":
                    continue

                decision = librarian_check_supersession(fact_text, neighbor_text)
                if not decision:
                    continue

                if decision.outcome == "A_supersedes_B":
                    cursor.execute(
                        "UPDATE atomic_facts SET temporal_status = 'historical' WHERE id = ?",
                        (neighbor_id,),
                    )
                    report["superseded"] += 1
                    # fact_id (A) is the current state; neighbor_id (B) is the past state
                    temporal_graph.add_relationship(
                        fact_id, "PRECEDED_BY", neighbor_id,
                        subject_name=fact_text[:80], object_name=neighbor_text[:80],
                        fact_ids=[neighbor_id], persist=False,
                    )
                    print(
                        f"[CONSOLIDATE] Text supersession: "
                        f"'{fact_text[:60]}' supersedes '{neighbor_text[:60]}'"
                    )
                elif decision.outcome == "B_supersedes_A":
                    cursor.execute(
                        "UPDATE atomic_facts SET temporal_status = 'historical' WHERE id = ?",
                        (fact_id,),
                    )
                    report["superseded"] += 1
                    # neighbor_id (B) is the current state; fact_id (A) is the past state
                    temporal_graph.add_relationship(
                        neighbor_id, "PRECEDED_BY", fact_id,
                        subject_name=neighbor_text[:80], object_name=fact_text[:80],
                        fact_ids=[fact_id], persist=False,
                    )
                    print(
                        f"[CONSOLIDATE] Text supersession: "
                        f"'{neighbor_text[:60]}' supersedes '{fact_text[:60]}'"
                    )
                    break
                elif decision.outcome == "contradiction":
                    protected_fact_ids.add(fact_id)
                    protected_fact_ids.add(neighbor_id)
                    report["flagged"].append({
                        "type": "contradiction",
                        "source": "text_based",
                        "fact_a": fact_text,
                        "fact_b": neighbor_text,
                    })
                    print(
                        f"[CONSOLIDATE] Text contradiction: "
                        f"'{fact_text[:60]}' vs '{neighbor_text[:60]}'"
                    )

        conn.commit()
        temporal_graph.write_graph()

        # ------------------------------------------------------------------
        # Phase 0: Exact-text dedup within atomic_facts
        # ------------------------------------------------------------------
        print("[CONSOLIDATE] Phase 0: Exact-text dedup in atomic_facts...")

        cursor.execute("""
            SELECT content, id, hit_count
            FROM atomic_facts
            WHERE content IN (
                SELECT content FROM atomic_facts GROUP BY content HAVING COUNT(*) > 1
            )
            ORDER BY content
        """)
        content_groups: dict[str, list] = {}
        for content, id_, hits in cursor.fetchall():
            content_groups.setdefault(content, []).append((id_, hits))

        for content, entries in content_groups.items():
            sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
            drop_fact_ids  = [e[0] for e in sorted_entries[1:]]

            if not drop_fact_ids:
                continue

            collection.delete(ids=drop_fact_ids)
            for cid in drop_fact_ids:
                knowledge_graph.remove_fact_reference(cid)
                temporal_graph.remove_fact_reference(cid)
            cursor.execute(
                f"DELETE FROM atomic_facts WHERE id IN ({','.join('?' * len(drop_fact_ids))})",
                drop_fact_ids
            )
            conn.commit()
            report["merged"] += len(drop_fact_ids)
            print(f"[CONSOLIDATE] Exact-text dedup: dropped {len(drop_fact_ids)} copy/copies of '{content[:60]}'")

        # ------------------------------------------------------------------
        # Phase 1: Prune stale atomic facts (never retrieved, older than N days)
        # ------------------------------------------------------------------
        print("[CONSOLIDATE] Phase 1: Pruning stale memories...")

        cutoff = (datetime.now() - timedelta(days=PRUNE_AGE_DAYS)).isoformat()
        cursor.execute(
            "SELECT id FROM atomic_facts WHERE hit_count = 0 AND created_at < ?",
            (cutoff,)
        )
        stale_fact_ids = [row[0] for row in cursor.fetchall()]

        if stale_fact_ids:
            collection.delete(ids=stale_fact_ids)
            cursor.execute(
                f"DELETE FROM atomic_facts WHERE id IN ({','.join('?' * len(stale_fact_ids))})",
                stale_fact_ids
            )
            conn.commit()
            for fact_id in stale_fact_ids:
                knowledge_graph.remove_fact_reference(fact_id)
                temporal_graph.remove_fact_reference(fact_id)
            report["pruned"] += len(stale_fact_ids)
            print(f"[CONSOLIDATE] Pruned {len(stale_fact_ids)} stale facts.")

        # ------------------------------------------------------------------
        # Phase 2: Near-duplicate detection → Librarian merge decision
        # ------------------------------------------------------------------
        print("[CONSOLIDATE] Phase 2: Detecting near-duplicates...")

        chroma_data = collection.get(include=["embeddings", "documents"])
        ids = chroma_data["ids"]
        docs = chroma_data["documents"]
        embeddings = chroma_data["embeddings"]

        merged_out = set()

        if len(ids) >= 2:
            emb_matrix = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            normalized = emb_matrix / np.maximum(norms, 1e-8)
            # Full pairwise cosine similarity in one matrix multiply — O(n²) but fast with numpy.
            # For very large collections (>5k facts), consider switching to approximate NN search.
            similarity_matrix = normalized @ normalized.T

            for i in range(len(ids)):
                if ids[i] in merged_out:
                    continue
                for j in range(i + 1, len(ids)):
                    if ids[j] in merged_out:
                        continue
                    sim = float(similarity_matrix[i, j])
                    if sim < DEDUP_SIMILARITY_THRESHOLD:
                        continue

                    # Skip contradiction-flagged facts and any fact already marked historical.
                    # Historical facts are archived versions — never dedup candidates.
                    if ids[i] in protected_fact_ids or ids[j] in protected_fact_ids:
                        continue
                    cursor.execute("SELECT temporal_status FROM atomic_facts WHERE id = ?", (ids[i],))
                    status_i = (cursor.fetchone() or ("current",))[0]
                    cursor.execute("SELECT temporal_status FROM atomic_facts WHERE id = ?", (ids[j],))
                    status_j = (cursor.fetchone() or ("current",))[0]
                    if "historical" in (status_i, status_j):
                        continue

                    if sim >= HIGH_SIM_DEDUP_THRESHOLD:
                        # Near-identical text — Librarian would likely return an empty merged_fact
                        # for two identical strings. Skip it; just drop the lower-hit copy.
                        cursor.execute("SELECT hit_count FROM atomic_facts WHERE id = ?", (ids[i],))
                        row_i = cursor.fetchone()
                        cursor.execute("SELECT hit_count FROM atomic_facts WHERE id = ?", (ids[j],))
                        row_j = cursor.fetchone()
                        hits_i = row_i[0] if row_i else 0
                        hits_j = row_j[0] if row_j else 0

                        drop_id = ids[j] if hits_i >= hits_j else ids[i]
                        collection.delete(ids=[drop_id])
                        cursor.execute("DELETE FROM atomic_facts WHERE id = ?", (drop_id,))
                        conn.commit()
                        knowledge_graph.remove_fact_reference(drop_id)
                        temporal_graph.remove_fact_reference(drop_id)

                        merged_out.add(ids[i])
                        merged_out.add(ids[j])
                        report["merged"] += 1
                        print(f"[CONSOLIDATE] Deduped exact duplicate (sim={sim:.3f}): '{docs[i][:70]}'")
                        break

                    # Medium similarity — ask the Librarian whether these are truly redundant
                    decision = librarian_should_merge(docs[i], docs[j])
                    if not decision or not decision.should_merge or not decision.merged_fact:
                        continue

                    # Remove both originals
                    collection.delete(ids=[ids[i], ids[j]])
                    cursor.execute("DELETE FROM atomic_facts WHERE id IN (?, ?)", (ids[i], ids[j]))
                    conn.commit()
                    knowledge_graph.remove_fact_reference(ids[i])
                    knowledge_graph.remove_fact_reference(ids[j])
                    temporal_graph.remove_fact_reference(ids[i])
                    temporal_graph.remove_fact_reference(ids[j])

                    # Add merged fact
                    merged_id = str(uuid.uuid4())
                    merged_vec = get_embedding(decision.merged_fact)
                    collection.add(
                        embeddings=[merged_vec],
                        documents=[decision.merged_fact],
                        ids=[merged_id]
                    )
                    cursor.execute(
                        "INSERT INTO atomic_facts (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
                        (merged_id, decision.merged_fact, now, now)
                    )
                    conn.commit()

                    merged_out.add(ids[i])
                    merged_out.add(ids[j])
                    report["merged"] += 1
                    print(
                        f"[CONSOLIDATE] Merged (sim={sim:.3f}):\n"
                        f"  A: {docs[i]}\n"
                        f"  B: {docs[j]}\n"
                        f"  → {decision.merged_fact}"
                    )
                    break  # Only one partner per fact per pass; re-run for further merges

        # ------------------------------------------------------------------
        # Phase 3: Split compound facts
        # ------------------------------------------------------------------
        # Fetch fresh snapshot — Phase 2 may have mutated the collection.
        print("[CONSOLIDATE] Phase 3: Splitting compound facts...")

        chroma_data = collection.get(include=["documents"])
        facts_to_check = [
            (id_, doc)
            for id_, doc in zip(chroma_data["ids"], chroma_data["documents"])
            if len(doc) >= COMPOUND_CHECK_MIN_CHARS
        ]

        for fact_id, fact_text in facts_to_check:
            decision = librarian_split_compound(fact_text)
            if not decision or not decision.is_compound or len(decision.split_facts) < 2:
                continue

            collection.delete(ids=[fact_id])
            cursor.execute("DELETE FROM atomic_facts WHERE id = ?", (fact_id,))
            knowledge_graph.remove_fact_reference(fact_id)
            temporal_graph.remove_fact_reference(fact_id)

            for split_fact in decision.split_facts:
                split_id = str(uuid.uuid4())
                split_vec = get_embedding(split_fact)
                collection.add(embeddings=[split_vec], documents=[split_fact], ids=[split_id])
                cursor.execute(
                    "INSERT INTO atomic_facts (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
                    (split_id, split_fact, now, now)
                )

            conn.commit()
            report["split"] += 1
            print(f"[CONSOLIDATE] Split into {len(decision.split_facts)} facts: '{fact_text[:60]}...'")

        # ------------------------------------------------------------------
        # KG cleanup: remove nodes that lost all edges (degree = 0)
        # ------------------------------------------------------------------
        orphaned = [n for n in list(knowledge_graph.G.nodes()) if knowledge_graph.G.degree(n) == 0]
        if orphaned:
            for node in orphaned:
                knowledge_graph.G.remove_node(node)
            knowledge_graph.write_graph()
            print(f"[CONSOLIDATE] Removed {len(orphaned)} orphaned KG nodes.")

        temp_orphaned = [
            n for n in list(temporal_graph.G.nodes()) if temporal_graph.G.degree(n) == 0
        ]
        if temp_orphaned:
            for node in temp_orphaned:
                temporal_graph.G.remove_node(node)
            temporal_graph.write_graph()
            print(f"[CONSOLIDATE] Removed {len(temp_orphaned)} orphaned temporal graph nodes.")

    print(f"[CONSOLIDATE] Done. {report}")
    return {"status": "success", "report": report}


@app.post("/memory/consolidate", status_code=202)
def consolidate_memories():
    """Enqueues a five-phase memory hygiene pass. Returns a task handle immediately.

    Poll GET /memory/task/{task_id} for status and the full consolidation report.
    """
    task_id = _create_task()
    _run_task_in_background(task_id, _consolidate_memories_sync)
    return {"task_id": task_id, "status": "pending"}
