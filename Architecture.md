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

#### LLM Client

`llm_client.py` owns all model loading and inference routing. A single `load_llm_client()` call at startup selects a backend and initializes it; every subsequent call uses `get_llm_client()`. Provider selection order (first match wins):

1. `LOCAL_MODEL: true` in `config.json` → local llama_cpp Qwen (explicit override)
2. `GOOGLE_API_KEY` env + `GOOGLE.MODEL_NAME` in config → Google Gemini
3. `OPENAI_API_KEY` + `OPENAI.MODEL_NAME` → OpenAI
4. `ANTHROPIC_API_KEY` + `ANTHROPIC.MODEL_NAME` → Anthropic (via OpenAI-compatible proxy)
5. `OLLAMA.BASE_URL` + `OLLAMA.MODEL_NAME` in config → Ollama (no key needed)
6. Fallback → local llama_cpp Qwen

Two backend classes share the same interface (`chat_json`, `chat_text`):
- `_LocalBackend` — llama_cpp, grammar-constrained JSON via the `"schema"` extension on `response_format`. Model variant (3B / 7B) and GPU layers read from `MUCH_RAM` / `USE_GPU` in `config.json`.
- `_OpenAICompatibleBackend` — openai SDK, `json_schema` response format (strict omitted — Pydantic optional fields are incompatible with OpenAI strict-mode client-side validation).

The BGE embedding model always runs locally via llama_cpp regardless of the Librarian backend.

#### Librarian

`librarian.py` owns all inference prompts and Pydantic output schemas. It has no llama_cpp dependency — all calls go through `get_llm_client()`. Functions:
- `process_memory_chunk()` — atomic fact extraction with temporal tagging
- `extract_entities_from_text()` — entity identification from search queries
- `librarian_should_merge()` — near-duplicate merge decisions
- `librarian_split_compound()` — compound fact splitting
- `librarian_check_supersession()` — supersession / contradiction classification (not yet wired into Phase 4)
- `extract_context_hint()` — subject + time_period extraction for cross-chunk pronoun resolution in `/learn`

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
- [x] Test infrastructure (pytest, conftest, 98 passing non-model tests)
- [x] Entity/predicate normalization at write time (title-case names, uppercase + synonym-map predicates)
- [x] `POST /memory/context` fast-path endpoint (no Librarian, vector-only, <100ms, `current` facts only)
- [x] Split `GET /memory/all` by type (`?type=raw|fact|entity`)
- [x] Layer 1 temporal resolution: `AtomicFact` model with `temporal_status` + `valid_period`; column migration in SQLite; Librarian prompt updated to tag tense at parse time
- [x] Phase 4 consolidation: structural supersession (IS/WAS edge pairs → mark WAS source facts `historical`) and contradiction detection (IS/IS_NOT edge pairs → flag for review). Phase 4 runs before Phase 2 to prevent embedding dedup from consuming opposing-predicate pairs.
- [x] **`POST /memory/learn` endpoint** — Splits large text into sentence-boundary chunks (`LEARN_CHUNK_SIZE = 5`, no overlap) and feeds each through the `/add` pipeline. Returns `{chunks_total, chunks_succeeded, facts_added, triples_added, errors}`. Context hint (`ContextHint`: subject + time_period) extracted from chunk 0 and prepended as `[CONTEXT: ...]` to subsequent chunks for cross-chunk pronoun resolution.
- [x] **Async memory operations** — `/add`, `/learn`, `/consolidate` return HTTP 202 + `{task_id, status:"pending"}` immediately. `GET /memory/task/{task_id}` polls status/result/error. Each background task opens its own SQLite connection (WAL mode). `_run_task_in_background` is monkeypatched to synchronous in tests.
- [x] **Unified LLM client** (`llm_client.py`) — Config-driven provider selection: local Qwen (llama_cpp) or any OpenAI-compatible cloud endpoint (Google Gemini, OpenAI, Ollama). `librarian.py` routes all inference through `get_llm_client()` with no llama_cpp dependency of its own.
- [x] **`librarian_check_supersession()` activation** — Phase 4b in consolidation: facts containing supersession keywords ("no longer", "used to", "formerly", etc.) are queried against ChromaDB neighbors; each pair is sent to `librarian_check_supersession()`; outcomes flip `temporal_status` to `historical` or flag contradictions with `source:"text_based"`. Pair dedup via `flagged_text_pairs` prevents redundant Librarian calls across passes.

#### Next / In Progress

- [x] **Source chunk linkage** — `source_chunk_id TEXT REFERENCES raw_chunks(id)` on `atomic_facts` and `entity_chunks (entity_id, chunk_id)` join table. Both populated at write time in `_add_memory_sync()`. Exposed on `/memory/search` and `/memory/all` responses. `_migrate_to_v3()` handles existing DBs.

- [x] **Tags / Groups** — Entity-only thematic clustering. `groups (id, name)` + `entity_groups (entity_id, group_id)` SQLite tables. `librarian_assign_groups()` called at write time in `_add_memory_sync()` for entities not yet in `entity_groups`. Surfaced on `/memory/all?type=entity` (`groups` list) and `/memory/search` (`entity_groups` map). Facts inherit group membership through their entities.

- [x] **Temporal relationship graph (Layer 2)** — `KnowledgeGraph/temporal_graph.json` using the same `KnowledgeRelationshipGraph` class. Nodes are fact UUIDs (state instances); edges are `PRECEDED_BY` with direction current → past. Populated by Phase 4 structural (WAS/IS edge pairs) and Phase 4b (text-based supersession via `librarian_check_supersession`). `/memory/search` returns `temporal_context: [{current_fact, preceded_by: [...]}]` built via `nx.descendants()` traversal with SQL read-time validation for dead endpoints. Fact deletions in consolidation Phases 0–3 propagate via `temporal_graph.remove_fact_reference()`. `DELETE /memory/clear` wipes the temporal graph. Known limitation: if a current-state node is merged by Phase 2, the PRECEDED_BY edge's source node lingers until the next orphan sweep (past-state cleanup is correct).
