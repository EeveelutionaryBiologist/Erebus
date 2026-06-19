import re
import json
import uuid
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from librarian import (
    load_librarian_model,
    process_memory_chunk,
    extract_entities_from_text,
    librarian_summarize,
    librarian_should_merge,
    librarian_split_compound,
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
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            hit_count   INTEGER DEFAULT 0,
            created_at  DATETIME,
            last_accessed DATETIME
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

# ==========================================
# 5. API ENDPOINTS
# ==========================================
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
        vector = get_embedding(fact)

        collection.add(embeddings=[vector], documents=[fact], ids=[fact_id])
        cursor.execute(
            "INSERT INTO atomic_facts (id, content, created_at, last_accessed) VALUES (?, ?, ?, ?)",
            (fact_id, fact, now, now)
        )
    sqlite_conn.commit()

    # 4. Save Triples to Knowledge Graph, linked to this batch's fact IDs
    for triple in processed_data.triples:
        subject_id = get_or_create_entity(triple.subject)
        object_id  = get_or_create_entity(triple.object)
        knowledge_graph.add_relationship(
            subject_id, triple.predicate, object_id,
            subject_name=triple.subject, object_name=triple.object,
            fact_ids=fact_ids_batch
        )
        print(f"  -> Graph Mapped: {triple.subject} [{triple.predicate}] {triple.object}")
    
    return {
        "status": "success", 
        "message": f"Added {len(processed_data.atomic_facts)} standalone facts and {len(processed_data.triples)} graph relations."
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

@app.get("/memory/all")
def get_all_memories():
    """Retrieves all records from all three tables in a unified flat list."""
    cursor = sqlite_conn.cursor()
    results = []

    cursor.execute("SELECT id, content, 0, created_at FROM raw_chunks ORDER BY created_at DESC")
    results += [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3], "record_type": "raw"} for r in cursor.fetchall()]

    cursor.execute("SELECT id, content, hit_count, created_at FROM atomic_facts ORDER BY created_at DESC")
    results += [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3], "record_type": "fact"} for r in cursor.fetchall()]

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
    knowledge_graph.G.clear()
    knowledge_graph.write_graph()
    
    return {"status": "success", "message": "All databases and graphs wiped clean."}


@app.post("/memory/consolidate")
def consolidate_memories():
    """
    Three-phase memory hygiene pass:
      Phase 1 – Prune: delete stale, never-retrieved atomic facts from ChromaDB + SQLite.
      Phase 2 – Merge: detect near-duplicate fact pairs via cosine similarity; ask the
                       Librarian to produce a merged fact and replace the originals.
      Phase 3 – Split: find compound facts that slipped through the write-time atomization
                       and break them into truly independent sentences.

    KG source tracking: each KG edge stores source_fact_ids. Deleting a fact calls
    knowledge_graph.remove_fact_reference(), which removes it from all edges and deletes
    edges whose source list becomes empty. Legacy edges (no source_fact_ids) are left
    in place and cleaned up by the degree=0 orphan sweep at the end of each pass.
    """
    report = {"pruned": 0, "merged": 0, "split": 0, "errors": []}
    cursor = sqlite_conn.cursor()
    now = datetime.now().isoformat()

    for _N in range(CONSOLIDATION_PASSES):
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
