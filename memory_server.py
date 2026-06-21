import re
import json
import uuid
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from librarian import (
    load_librarian_model,
    process_memory_chunk,
    extract_entities_from_text,
    extract_context_hint,
    librarian_summarize,
    librarian_should_merge,
    librarian_split_compound,
    librarian_check_supersession,
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
    
    # Load background Librarian
    load_librarian_model()
    _migrate_to_v2()

def get_embedding(text: str) -> list[float]:
    response = embedder.create_embedding(text)
    return response["data"][0]["embedding"]

# Initialize Knowledge Graph
knowledge_graph = KnowledgeRelationshipGraph(str(GRAPH_DIR / "knowledge_graph.json"))

# ==========================================
# 3. DATABASE INITIALIZATION
# ==========================================
def init_sqlite():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
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

def get_or_create_entity(name: str) -> str:
    """Returns the entity UUID for `name`, inserting a new row if it doesn't exist."""
    cursor = sqlite_conn.cursor()
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
    sqlite_conn.commit()
    return entity_id

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

# ==========================================
# 5. API ENDPOINTS
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

@app.post("/memory/add")
def add_memory(memory: MemoryInput):
    """Processes raw text via Librarian, saving Atomic facts to ChromaDB and Triples to Graph."""
    now = datetime.now().isoformat()
    cursor = sqlite_conn.cursor()
    
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
    sqlite_conn.commit()

    # 3. Save Atomic Facts to ChromaDB & SQLite
    fact_ids_batch = []
    for fact in processed_data.atomic_facts:
        fact_id = str(uuid.uuid4())
        fact_ids_batch.append(fact_id)
        vector = get_embedding(fact.text)

        collection.add(embeddings=[vector], documents=[fact.text], ids=[fact_id])
        cursor.execute(
            "INSERT INTO atomic_facts "
            "(id, content, temporal_status, valid_period, created_at, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fact_id, fact.text, fact.temporal_status, fact.valid_period, now, now)
        )
    sqlite_conn.commit()

    # 4. Save Triples to Knowledge Graph, linked to this batch's fact IDs.
    # persist=False defers the disk write; we flush once after the loop.
    for triple in processed_data.triples:
        subj = normalize_entity_name(triple.subject)
        obj  = normalize_entity_name(triple.object)
        pred = normalize_predicate(triple.predicate)
        subject_id = get_or_create_entity(subj)
        object_id  = get_or_create_entity(obj)
        knowledge_graph.add_relationship(
            subject_id, pred, object_id,
            subject_name=subj, object_name=obj,
            fact_ids=fact_ids_batch,
            persist=False,
        )
        print(f"  -> Graph Mapped: {subj} [{pred}] {obj}")
    if processed_data.triples:
        knowledge_graph.write_graph()
    
    return {
        "status": "success",
        "message": f"Added {len(processed_data.atomic_facts)} standalone facts and {len(processed_data.triples)} graph relations.",
        "facts_added": len(processed_data.atomic_facts),
        "triples_added": len(processed_data.triples),
    }


@app.post("/memory/learn")
def learn_from_source(memory: MemoryInput):
    """Splits large text into sentence-boundary chunks and feeds each through /memory/add.

    For multi-chunk inputs, the first chunk is passed to extract_context_hint() to produce a
    short [CONTEXT: subject, time_period] prefix. This prefix is prepended to all subsequent
    chunks before Librarian processing, grounding pronoun resolution and temporal tagging
    across chunk boundaries. Works best for single-subject texts (biographies, diaries).
    """
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
            result = add_memory(MemoryInput(text=chunk_text))
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
            cursor.execute("SELECT content, hit_count FROM atomic_facts WHERE id = ?", (mem_id,))
            row = cursor.fetchone()
            if row:
                final_results.append({"text": row[0], "hit_count": row[1]})
    sqlite_conn.commit()

    # --- 2. GRAPH RETRIEVAL ---
    relation_facts = []
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
    sqlite_conn.commit()
    
    summarized_context = ""
    if relation_facts:
        unique_facts = list(set(relation_facts)) # Deduplicate facts before summary
        # summarized_context = librarian_summarize(unique_facts) # <--- This eats quite a lot of power...
        summarized_context = "\n".join(unique_facts)
        
    return {
        "results": final_results,
        "relational_context": summarized_context
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
        cursor.execute("SELECT id, content, hit_count, created_at FROM atomic_facts ORDER BY created_at DESC")
        results += [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3], "record_type": "fact"} for r in cursor.fetchall()]

    if type is None or type == "entity":
        cursor.execute("SELECT id, canonical_name, hit_count, created_at FROM entities ORDER BY created_at DESC")
        results += [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3], "record_type": "entity"} for r in cursor.fetchall()]

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
    
    # 3. Clear Graph
    knowledge_graph.clear()
    
    return {"status": "success", "message": "All databases and graphs wiped clean."}


@app.post("/memory/consolidate")
def consolidate_memories():
    """
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
    cursor = sqlite_conn.cursor()
    now = datetime.now().isoformat()

    # Tracks contradiction pairs already added to report["flagged"] so multi-pass doesn't duplicate.
    flagged_pairs: set[tuple[str, str, str, str]] = set()
    # Fact IDs involved in a detected contradiction — protected from Phase 2 dedup since they
    # are intentionally distinct facts, not duplicates.
    protected_fact_ids: set[str] = set()

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

        sqlite_conn.commit()

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
            cursor.execute(
                f"DELETE FROM atomic_facts WHERE id IN ({','.join('?' * len(drop_fact_ids))})",
                drop_fact_ids
            )
            sqlite_conn.commit()
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
            sqlite_conn.commit()
            for fact_id in stale_fact_ids:
                knowledge_graph.remove_fact_reference(fact_id)
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
                        sqlite_conn.commit()
                        knowledge_graph.remove_fact_reference(drop_id)

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
                    sqlite_conn.commit()
                    knowledge_graph.remove_fact_reference(ids[i])
                    knowledge_graph.remove_fact_reference(ids[j])

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
                    sqlite_conn.commit()

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

            for split_fact in decision.split_facts:
                split_id = str(uuid.uuid4())
                split_vec = get_embedding(split_fact)
                collection.add(embeddings=[split_vec], documents=[split_fact], ids=[split_id])
                cursor.execute(
                    "INSERT INTO atomic_facts (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
                    (split_id, split_fact, now, now)
                )

            sqlite_conn.commit()
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

    print(f"[CONSOLIDATE] Done. {report}")
    return {"status": "success", "report": report}
