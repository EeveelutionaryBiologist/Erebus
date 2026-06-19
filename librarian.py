import json
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel, Field

# TODO: Move this to a config file
MUCH_RAM = True

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Embedding"

if MUCH_RAM:
    LIBRARIAN_MODEL_PATH = MODEL_DIR / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
else:
    LIBRARIAN_MODEL_PATH = MODEL_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"


# Global variable for the model
librarian_llm = None

# --- PYDANTIC SCHEMAS ---
class Entity(BaseModel):
    name: str

class EntityExtraction(BaseModel):
    entities: list[Entity]

class KnowledgeTriple(BaseModel):
    subject: str = Field(description="The main entity (e.g., 'Hailey')")
    predicate: str = Field(description="The relationship (e.g., 'HAS', 'IS', 'MOTHER_OF')")
    object: str = Field(description="The target entity (e.g., 'Mochi')")

class MemoryProcessing(BaseModel):
    atomic_facts: list[str] = Field(
        description="A list of standalone, independent sentences extracted from the text. All pronouns must be replaced with the actual entity names so the sentence makes sense in isolation."
    )
    triples: list[KnowledgeTriple]

class MergeDecision(BaseModel):
    should_merge: bool
    merged_fact: str = Field(
        default="",
        description="The single merged fact if should_merge is True. Empty string if False."
    )

class SplitDecision(BaseModel):
    is_compound: bool
    split_facts: list[str] = Field(
        default_factory=list,
        description="The atomic sub-facts if is_compound is True. Empty list if False."
    )

# --- FUNCTIONS ---
def load_librarian_model():
    """Downloads and loads the local Librarian model permanently into RAM."""
    global librarian_llm
    if not LIBRARIAN_MODEL_PATH.exists():
        print("[SYSTEM] Downloading librarian background model...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MUCH_RAM:
            snapshot_download(
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                local_dir=MODEL_DIR,
                allow_patterns=["qwen2.5-7b-instruct-q4_k_m*"]
            )
        else:
            hf_hub_download(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                local_dir=MODEL_DIR,
            )
    print("[SYSTEM] Initializing Llama.cpp Librarian Model in RAM...")
    librarian_llm = Llama(
        model_path=str(LIBRARIAN_MODEL_PATH), 
        n_ctx=4096,
        n_gpu_layers=0,       # Force CPU/RAM
        use_mlock=True,       # Prevent OS swapping
        verbose=False,
        chat_format="chatml"  # Required for Qwen models
    )

def process_memory_chunk(text: str) -> MemoryProcessing:
    """Extracts atomic facts (for ChromaDB) AND triples (for the Graph) simultaneously."""
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded.")
        
    system_prompt = (
        "You are an advanced data extraction AI. You have two tasks:\n"
        "1. Extract 'atomic_facts': Break the text into independent, single-fact sentences. "
        "CRITICAL: Resolve all pronouns. 'She has cats' MUST become 'Hailey has cats'.\n"
        "2. Extract 'triples': Create Subject-Predicate-Object relations for the knowledge graph."
    )
    
    print(f"[LIBRARIAN] Processing chunk for DBs: '{text[:50]}...'")
    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Process this text: '{text}'"}
        ],
        response_format={
            "type": "json_object",
            "schema": MemoryProcessing.model_json_schema()
        },
        temperature=0.1
    )
    
    try:
        output_str = response['choices'][0]['message']['content']
        extracted_data = json.loads(output_str)
        return MemoryProcessing(**extracted_data)
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Failed to process memory: {e}")
        return None

def extract_entities_from_text(text: str) -> EntityExtraction:
    """Pulls entities from user queries so we know which nodes to search in the Graph."""
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded.")
        
    system_prompt = "Extract key nouns, proper nouns, and entities from the query."
    
    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract entities from: '{text}'"}
        ],
        response_format={
            "type": "json_object",
            "schema": EntityExtraction.model_json_schema()
        },
        temperature=0.1
    )
    
    try:
        output_str = response['choices'][0]['message']['content']
        return EntityExtraction(**json.loads(output_str))
    except Exception:
        return None

def librarian_should_merge(fact_a: str, fact_b: str) -> MergeDecision | None:
    """Returns a merge decision for two semantically similar facts."""
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded.")

    system_prompt = (
        "You are a memory deduplication engine. Given two facts, decide if they are "
        "semantically equivalent or if one is a strict subset of the other. "
        "If yes, write a single merged fact that preserves the most specific information from both. "
        "If they describe genuinely different things, do NOT merge."
    )

    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Fact A: "{fact_a}"\nFact B: "{fact_b}"'}
        ],
        response_format={"type": "json_object", "schema": MergeDecision.model_json_schema()},
        temperature=0.1
    )

    try:
        return MergeDecision(**json.loads(response['choices'][0]['message']['content']))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Merge decision failed: {e}")
        return None


def librarian_split_compound(fact: str) -> SplitDecision | None:
    """Returns a split decision for a potentially compound fact."""
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded.")

    system_prompt = (
        "You are a memory atomization engine. A fact is 'compound' if it contains two or more "
        "independent pieces of information that would each make sense as a standalone sentence. "
        "If compound, split it into the smallest possible independent facts. "
        "Resolve all pronouns in each split so they make sense in isolation. "
        "If the statement is already a single atomic fact, set is_compound=false."
    )

    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Is this compound? "{fact}"'}
        ],
        response_format={"type": "json_object", "schema": SplitDecision.model_json_schema()},
        temperature=0.1
    )

    try:
        return SplitDecision(**json.loads(response['choices'][0]['message']['content']))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Split decision failed: {e}")
        return None


def librarian_summarize(facts: list[str]) -> str:
    """Takes raw 'A [OWNS] B' facts and makes them a readable string for the main agent."""
    if not facts:
        return ""
        
    facts_text = "\n".join(facts)
    system_prompt = "You are a concise AI. Combine these relational facts into a brief, human-readable summary."
    
    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize these facts:\n{facts_text}"}
        ],
        temperature=0.3
    )
    
    return response['choices'][0]['message']['content']

def memory_consolidation_routine():
    """
    Consolidation orchestration lives in memory_server.py as POST /memory/consolidate,
    since it requires direct access to SQLite, ChromaDB, and the knowledge graph.

    This module provides the LLM primitives the endpoint calls:
      - librarian_should_merge()   → deduplication: should two similar facts become one?
      - librarian_split_compound() → atomization: does a fact contain multiple independent claims?

    To trigger consolidation: POST http://localhost:8000/memory/consolidate
    Or use the /consolidate system command from agent.py (see parse_system_prompt).
    """
    pass
