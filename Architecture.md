# Erebus

LLM locally managed persistent memory system.

### Purpose

This will serve as the memory backend of the - also under development - Nyxx Agentic Chat interface. Originally one project,
it was decided that it would serve its purpose better by being a self-contained entity that can serve as a dynamic memory backend
that can be communicated with via FastAPI endpoints (as specified in the memory_server.py).


### Overall structure

#### Databases

Three SQLite tables in `DB/metadata.db`:
- **`raw_chunks`** — verbatim text submitted via `/memory/add`
- **`atomic_facts`** — broken-down statements with `temporal_status` (`current` / `historical` / `uncertain`) and optional `valid_period`
- **`entities`** — entity names from the knowledge graph with hit counts and aliases

ChromaDB at `DB/chromadb/` indexes atomic facts for vector search. All RAG querying uses a local embedding model (`bge-base-en-v1.5-f16.gguf`).

IMPORTANT NOTE: Any and all data currently held in databases / knowledge graphs is irrelevant and just for test purposes. Can be deleted / reinitialized whenever.

#### Knowledge Graph

`KnowledgeGraph/knowledge_graph.json` — a NetworkX `MultiDiGraph` where:
- Node keys are entity UUIDs; human-readable names are node attributes
- Multiple predicates between the same entity pair are supported as separate edges
- Each edge carries `source_fact_ids` linking it to the atomic facts that produced it
- `_fact_edge_index` (`{fact_id: [(subject_id, object_id, edge_key)]}`) enables O(1) fact-deletion propagation

A second temporal graph (`KnowledgeGraph/temporal_graph.json`) is planned (see Roadmap below).

#### Librarian

LLM-based routine (`librarian.py`) that handles:
- Atomic fact extraction with temporal tagging (`process_memory_chunk()`)
- Entity identification from queries (`extract_entities_from_text()`)
- Near-duplicate merge decisions (`librarian_should_merge()`)
- Compound fact splitting (`librarian_split_compound()`)
- Supersession / contradiction classification (`librarian_check_supersession()`, not yet wired into Phase 4)

Controlled by `MUCH_RAM` flag: `True` → Qwen2.5-7B-Instruct (two GGUF shards); `False` → Qwen2.5-3B-Instruct.

#### Notable API Endpoints

##### POST /memory/add
Process text → atomic facts (ChromaDB + SQLite) + entity triples (knowledge graph). Entity names and predicates are normalized at write time (title-case names, uppercase predicates, synonym map).

##### POST /memory/context
Fast-path retrieval for Nyxx. No Librarian — vector search (top_k=3) + depth-1 graph lookup via a regex tokenizer. Returns only `current` facts (historical facts are excluded). Designed for <100ms.

##### POST /memory/search
Deep retrieval. Vector search + Librarian-powered entity extraction + depth-1 KG traversal. Returns both `current` and `historical` facts.

##### POST /memory/consolidate
Five-phase hygiene pass (runs `CONSOLIDATION_PASSES = 2` times):
1. **Phase 4 (runs first)** — Structural supersession/contradiction via KG: marks WAS/HAD source facts as `historical` when a IS/HAS edge exists for the same entity pair; flags IS/IS_NOT pairs as contradictions for review. Protected facts and historical facts are immune to Phase 2 dedup.
2. **Phase 0** — Exact-text dedup within `atomic_facts`.
3. **Phase 1** — Prune stale facts (zero hits, older than `PRUNE_AGE_DAYS`).
4. **Phase 2** — Near-duplicate merge via cosine similarity; Librarian decides on merge text. Skips historical facts and contradiction-protected facts.
5. **Phase 3** — Split compound facts into atomic sentences.

##### GET /memory/all
Tabular view of all stored elements. Optional `?type=raw|fact|entity` filter.

##### DELETE /memory/clear
Wipes ChromaDB collection, all SQLite tables, and the knowledge graph.


### Roadmap

#### Completed

- [x] Entity table in SQLite (`id, canonical_name, aliases, hit_count, created_at, last_accessed`)
- [x] Three-table DB split (`raw_chunks` / `atomic_facts` / `entities`) with v2 migration
- [x] UUID node keys in KG (linked to entities table)
- [x] O(1) fact→edge index (`_fact_edge_index`)
- [x] MultiDiGraph (multiple predicates per entity pair)
- [x] Graph write batching (`persist=False` + `write_graph()` flush at end of `/memory/add`)
- [x] Test infrastructure (pytest, conftest, 71 passing non-model tests)
- [x] Entity/predicate normalization at write time (title-case names, uppercase + synonym-map predicates)
- [x] `POST /memory/context` fast-path endpoint (no Librarian, vector-only, <100ms, `current` facts only)
- [x] Split `GET /memory/all` by type (`?type=raw|fact|entity`)
- [x] Layer 1 temporal resolution: `AtomicFact` model with `temporal_status` + `valid_period`; column migration in SQLite; Librarian prompt updated to tag tense at parse time
- [x] Phase 4 consolidation: structural supersession (IS/WAS edge pairs → mark WAS source facts `historical`) and contradiction detection (IS/IS_NOT edge pairs → flag for review). Phase 4 runs before Phase 2 to prevent embedding dedup from consuming opposing-predicate pairs.

#### Next / In Progress

- [ ] **Tags / Groups** — Thematic clusters that can be applied to entities, facts, or both. An entity can belong to multiple groups (e.g., "University", "Friends"). During `/memory/add`, a Librarian call should decide whether the entity fits an existing group or warrants a new one. Groups should be persisted (SQLite `groups` table + a many-to-many `entity_groups` join table) and surfaced on `/memory/search` and `/memory/context` results. Open questions: should facts be group-tagged independently of their entities, or inherit group membership from their entities?

- [ ] **Temporal relationship graph (Layer 2)** — A second `KnowledgeGraph/temporal_graph.json` using the same `KnowledgeRelationshipGraph` class. Nodes are state-instances (not bare entities); edges are `CAUSED`, `PRECEDED_BY`, `CONCURRENT_WITH`. Populated by Phase 4 supersession detection results. Enables `nx.ancestors()` / `nx.descendants()` traversal in `/memory/search` for transitive causal context ("if A caused B and B caused C, then C was implicitly caused by A").

- [ ] **Async consolidation** — `POST /memory/consolidate` is synchronous and will block/time out on large collections. Move to FastAPI `BackgroundTask` with a status endpoint.

- [ ] **`librarian_check_supersession()` activation** — The function exists in `librarian.py` but Phase 4 currently uses only structural graph detection. Future: call it for same-predicate pairs where the fact text contains supersession language ("no longer", "used to") but no opposing predicate exists in the KG.

- [x] **`POST /learn` endpoint** — Splits large text into sentence-boundary chunks (`LEARN_CHUNK_SIZE = 5`, no overlap) and feeds each through the `/add` pipeline. Returns `{chunks_total, chunks_succeeded, facts_added, triples_added, errors}`. Partial success is reported as `status: "partial"` with per-chunk error details. No overlap is used to avoid duplicate ingestion; cross-sentence pronoun resolution within a chunk is handled by Librarian context.

- [ ] **Async memory operations** — `/add`, `/learn`, and `/consolidate` all block the server during Librarian inference (30s+ for large inputs or consolidation passes). All three should be moved to `BackgroundTask` (or a task queue) so the server remains responsive. Each should return a task ID immediately, with a `GET /memory/task/{id}` status endpoint to poll for completion.
