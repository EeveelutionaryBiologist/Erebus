import os
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
    _backfill_record_types()

def get_embedding(text: str) -> list[float]:
    response = embedder.create_embedding(text)
    return response["data"][0]["embedding"]

# Initialize Knowledge Graph
knowledge_graph = KnowledgeRelationshipGraph(str(MEMORY_DIR / "knowledge_graph.json"))

# ==========================================
# 3. DATABASE INITIALIZATION
# ==========================================
def init_sqlite():
    # check_same_thread=False is critical to prevent FastAPI threading crashes
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            hit_count INTEGER DEFAULT 0,
            created_at DATETIME,
            last_accessed DATETIME
        )
    ''')
    conn.commit()

    # Migration: add record_type column if this is an existing database
    try:
        cursor.execute("ALTER TABLE memories ADD COLUMN record_type TEXT DEFAULT 'fact'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

    return conn

sqlite_conn = init_sqlite()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="nyxx_memory")

def _backfill_record_types():
    """
    One-time migration for databases created before the record_type column existed.
    Any SQLite record whose ID is not in ChromaDB is a raw chunk — tag it 'raw'.
    Runs at startup; safe to call multiple times (no-op if already tagged).
    """
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id FROM memories WHERE record_type = 'fact' OR record_type IS NULL")
    candidates = [row[0] for row in cursor.fetchall()]
    if not candidates:
        return

    chroma_data = collection.get(include=[])
    chroma_ids = set(chroma_data["ids"])

    sqlite_only = [id_ for id_ in candidates if id_ not in chroma_ids]
    if sqlite_only:
        cursor.execute(
            f"UPDATE memories SET record_type='raw' WHERE id IN ({','.join('?' * len(sqlite_only))})",
            sqlite_only
        )
        sqlite_conn.commit()
        print(f"[SYSTEM] Backfilled {len(sqlite_only)} existing records as record_type='raw'.")

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
        "INSERT INTO memories (id, content, created_at, last_accessed, record_type) VALUES (?, ?, ?, ?, ?)",
        (raw_id, memory.text, now, now, 'raw')
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
            "INSERT INTO memories (id, content, created_at, last_accessed, record_type) VALUES (?, ?, ?, ?, ?)",
            (fact_id, fact, now, now, 'fact')
        )
    sqlite_conn.commit()

    # 4. Save Triples to Knowledge Graph, linked to this batch's fact IDs
    for triple in processed_data.triples:
        knowledge_graph.add_relationship(
            triple.subject, triple.predicate, triple.object,
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
            cursor.execute("""
                UPDATE memories 
                SET hit_count = hit_count + 1, last_accessed = ? 
                WHERE id = ?
            """, (now, mem_id))
            sqlite_conn.commit()
            
            cursor.execute("SELECT content, hit_count FROM memories WHERE id = ?", (mem_id,))
            row = cursor.fetchone()
            if row:
                final_results.append({
                    "text": row[0],
                    "hit_count": row[1]
                })

    # --- 2. GRAPH RETRIEVAL ---
    relation_facts = [] # Initialize safely
    extracted = extract_entities_from_text(search.query)
    
    if extracted and hasattr(extracted, 'entities'):
        for entity in extracted.entities:
            facts = knowledge_graph.retrieve_relationships(entity.name, depth=1)
            if facts:
                relation_facts.extend(facts)
    
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
    """Retrieves all memories currently stored in the SQLite database."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, content, hit_count, created_at, record_type FROM memories ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return {"results": [{"id": r[0], "text": r[1], "hit_count": r[2], "created_at": r[3], "record_type": r[4]} for r in rows]}

@app.delete("/memory/clear")
def clear_all_memories():
    """Wipes ChromaDB, SQLite, and the Knowledge Graph completely."""
    # 1. Clear SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute("DELETE FROM memories")
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
        # Phase 0: Exact-text dedup across SQLite
        # ------------------------------------------------------------------
        # Uses record_type to distinguish atomic facts (in ChromaDB) from raw chunks
        # (SQLite-only). Prefers keeping the 'fact' record; drops raw-chunk duplicates
        # and any extra fact copies.
        print("[CONSOLIDATE] Phase 0: Exact-text dedup in SQLite...")

        cursor.execute("""
            SELECT content, id, COALESCE(record_type, 'fact') as rt, hit_count
            FROM memories
            WHERE content IN (
                SELECT content FROM memories GROUP BY content HAVING COUNT(*) > 1
            )
            ORDER BY content
        """)
        content_groups: dict[str, list] = {}
        for content, id_, rt, hits in cursor.fetchall():
            content_groups.setdefault(content, []).append((id_, rt, hits))

        for content, entries in content_groups.items():
            facts = [(id_, hits) for id_, rt, hits in entries if rt == 'fact']
            raws  = [id_ for id_, rt, hits in entries if rt != 'fact']

            if facts:
                facts_sorted = sorted(facts, key=lambda x: x[1], reverse=True)
                keep_id       = facts_sorted[0][0]
                drop_fact_ids = [e[0] for e in facts_sorted[1:]]
                drop_ids      = drop_fact_ids + raws
            else:
                all_sorted    = sorted(entries, key=lambda x: x[2], reverse=True)
                keep_id       = all_sorted[0][0]
                drop_ids      = [e[0] for e in all_sorted[1:]]
                drop_fact_ids = []

            if not drop_ids:
                continue

            if drop_fact_ids:
                collection.delete(ids=drop_fact_ids)
                for cid in drop_fact_ids:
                    knowledge_graph.remove_fact_reference(cid)

            cursor.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' * len(drop_ids))})",
                drop_ids
            )
            sqlite_conn.commit()
            report["merged"] += len(drop_ids)
            print(f"[CONSOLIDATE] Exact-text dedup: dropped {len(drop_ids)} copy/copies of '{content[:60]}'")

        # ------------------------------------------------------------------
        # Phase 1: Prune stale atomic facts (never retrieved, older than N days)
        # ------------------------------------------------------------------
        # record_type='fact' targets only ChromaDB-backed entries; raw chunks are
        # excluded since their hit_count is always 0 but they should not be pruned.
        print("[CONSOLIDATE] Phase 1: Pruning stale memories...")

        cutoff = (datetime.now() - timedelta(days=PRUNE_AGE_DAYS)).isoformat()
        cursor.execute(
            "SELECT id FROM memories WHERE hit_count = 0 AND created_at < ? AND record_type = 'fact'",
            (cutoff,)
        )
        stale_fact_ids = [row[0] for row in cursor.fetchall()]

        if stale_fact_ids:
            collection.delete(ids=stale_fact_ids)
            cursor.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' * len(stale_fact_ids))})",
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
                        cursor.execute("SELECT hit_count FROM memories WHERE id = ?", (ids[i],))
                        row_i = cursor.fetchone()
                        cursor.execute("SELECT hit_count FROM memories WHERE id = ?", (ids[j],))
                        row_j = cursor.fetchone()
                        hits_i = row_i[0] if row_i else 0
                        hits_j = row_j[0] if row_j else 0

                        drop_id = ids[j] if hits_i >= hits_j else ids[i]
                        collection.delete(ids=[drop_id])
                        cursor.execute("DELETE FROM memories WHERE id = ?", (drop_id,))
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
                    cursor.execute("DELETE FROM memories WHERE id IN (?, ?)", (ids[i], ids[j]))
                    sqlite_conn.commit()
                    knowledge_graph.remove_fact_reference(ids[i])
                    knowledge_graph.remove_fact_reference(ids[j])

                    # Add merged fact (tagged as 'fact' so it's included in future dedup passes)
                    merged_id = str(uuid.uuid4())
                    merged_vec = get_embedding(decision.merged_fact)
                    collection.add(
                        embeddings=[merged_vec],
                        documents=[decision.merged_fact],
                        ids=[merged_id]
                    )
                    cursor.execute(
                        "INSERT INTO memories (id, content, created_at, last_accessed, record_type) VALUES (?, ?, ?, ?, ?)",
                        (merged_id, decision.merged_fact, now, now, 'fact')
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
            cursor.execute("DELETE FROM memories WHERE id = ?", (fact_id,))
            knowledge_graph.remove_fact_reference(fact_id)

            for split_fact in decision.split_facts:
                split_id = str(uuid.uuid4())
                split_vec = get_embedding(split_fact)
                collection.add(embeddings=[split_vec], documents=[split_fact], ids=[split_id])
                cursor.execute(
                    "INSERT INTO memories (id, content, created_at, last_accessed, record_type) VALUES (?, ?, ?, ?, ?)",
                    (split_id, split_fact, now, now, 'fact')
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
