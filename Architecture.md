# Erebus

LLM locally managed persistent memory system.

### Purpose

This will serve as the memory backend of the - also under development - Nyxx Agentic Chat interface. Originally one project, 
it was decided that it would serve it's purpose better by being a self-contained entity that can serve as a dynamic memory backend
that can be communicated with via FastAPI endpoints (as specified in the memory_server.py).


### Overall planned structure 

#### Databases
Currently, all raw memory strings and atomic facts are stored in one unified sqlite data base handled by ChromaDB. The plan is 
to split up this arhcitecture into three distinct db tables/collections: Raw memory (unedited text strings that were committed to memory via API), 
Atomic Facts (broken down statements of type [A] likes [B] and Entities (Literally just names of entities in the knowledge graph and their 
respective time of addition, hits_count, etc.). 
RAG querying of the data base is currently handled via a local embedding model. As the goal is an entirely self-hosted system, this is to stay that way.

#### Knowledge Graph 
There is currently a single entity relationship graph handled via a corresponding class in knowledge_graph.py. While sufficient for the time being,
perspectively, we want to add a graph representing temporal relationships of entities. For the most imminent changes, we however first want to
establish a clear key-based linkage between Graph nodes (Entities) and the respective table, as well as for the edges and corresponding atomic facts.

#### Librarian
LLM-based routine that handles:
- Entity identification
- Atomic facts parsing
- memory consolidation
- summarizing

#### Mental Notes
The librarian might adome point write high-level summaries of certain entity relationships, contextual descriptions. Not sure about good trigger conditions here, yet.

#### Notable API endpoints

Entries marked with a [!] are yet to be implemented. All endpoints need to be refactored in some capacity to 
adjust to architectural changes.

##### POST /memory/add
Insert a raw text string into memory, break up into atomic facts and entity relationships and update the corresponding DB and knowledge graph.

##### POST /memory/context [!]
Immediate context enrichment of a query based on data base and entity relationships. This should be as fast as possible as to not lag the 
frontend model.

##### POST /memory/search
Takes a query and retrieves a more in-depth contextual enrichment of the subject at hand. Primarily this should be handled via a higher depth parameter
for the knowledge graph search, but algorithmic suggestions are welcome on how to enhance results here. Quality/Depth > Speed.

##### POST /memory/consolidate
The core cleanup routine of the librarian, running periodically. The goal is to remove orphaned nodes, merge/consolidate redundant information and resolve potential contradictions. 

##### POST /memory/all

Retrieves a Tabular view of all stored elements in the data base (for the user/testing). Should probably be split up into individual views for Raw, Atomic facts, Entities.

##### POST /memory/clear

Wipes the data base. This is a destructive action.


### Immediate ToDo

#### Bug: Knowledge Graph path mismatch
`memory_server.py:74` constructs the graph path as `MEMORY_DIR / "knowledge_graph.json"` → `DB/knowledge_graph.json`.
`GRAPH_DIR` (`KnowledgeGraph/`) is created at startup but never used. The graph lands in the wrong directory.
Fix: change line 74 to use `GRAPH_DIR / "knowledge_graph.json"`.

---

#### Architectural suggestions

**Aligned with roadmap (near-term)**

1. **Entity table** — Entities currently exist only as bare string node keys in NetworkX with no metadata. Before the linkage refactor, add a dedicated `entities` SQLite table: `id` (UUID), `canonical_name`, `aliases` (JSON list), `hit_count`, `created_at`. This is the prerequisite for everything below.

2. **Key-based node↔entity and edge↔fact linkage** — After the entity table exists, switch KG node keys from raw name strings to entity UUIDs (store the human-readable name as a node attribute). Edges already track `source_fact_ids`; the reverse (given a fact, which edges reference it?) currently requires a full edge scan in `remove_fact_reference()`. A simple `fact_id → [edge_keys]` index on the entity table would make deletions O(1) instead of O(E).

3. **DB three-table split** — Split the unified `memories` table into `raw_chunks` and `atomic_facts` (and `entities` from item 1). All the `WHERE record_type = 'fact'` filters in consolidation go away; the tables themselves carry the distinction. Requires a one-time migration.

4. **Entity/predicate normalization at write time** — KG nodes are currently created from raw LLM output, so "hailey", "Hailey", and "she" can all become separate nodes. The case-insensitive fallback in `retrieve_relationships()` is a retrieval patch, not a fix. The Librarian should be prompted to emit a canonical entity form (e.g. title-case proper noun), and `add_relationship()` should normalize before inserting to avoid duplicates at rest. Similarly, predicates arrive as arbitrary strings ("HAS", "has a", "possesses") — canonicalize to uppercase with a small lookup table for common synonyms.

5. **`MultiDiGraph` for multi-predicate edges** — The current `DiGraph` allows only one edge per entity pair (first writer wins). Two entities can have more than one relationship (e.g. A WORKS_AT B and A OWNS B). Switching to `nx.MultiDiGraph` is a small change but needs to happen before the graph grows large.

6. **`POST /memory/context`** — The planned fast-path endpoint. Should skip the Librarian entirely (no entity extraction) and run vector search only with a small `top_k` (3–5). Raw graph depth-1 lookup on exact-match entities from a simple noun-phrase regex is fast enough. Return in <100ms. This is the path Nyxx hits on every message; `/memory/search` remains the deep path.

7. **Split `/memory/all`** — Either per-type endpoints (`/memory/all/raw`, `/memory/all/facts`, `/memory/all/entities`) or a `?type=` query param. The current mixed view makes debugging harder as the DB grows.

---

**Later / optimizations**

- **Async consolidation** — The `POST /memory/consolidate` endpoint is synchronous and blocks until all Librarian calls complete. For large collections, it will time out. Move to a FastAPI `BackgroundTask` with a status endpoint.
- **Graph write batching** — `add_relationship()` serializes the full graph to disk on every call. For bulk ingestion (many facts per `/add`), batch the writes and flush once at the end.
- **Contradiction detection** — The consolidation routine currently merges near-duplicates but has no mechanism to detect contradictions (A IS_ALLERGIC_TO X vs. A EATS X). A future consolidation phase could flag edge pairs with semantically opposing predicates for Librarian review.
- **Temporal relationship graph** — As noted in the overall plan: a second graph tracking when relationships were established. Low priority until the primary graph is stable.