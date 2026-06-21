import json
from pathlib import Path
from typing import Literal
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel, Field, field_validator

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

class AtomicFact(BaseModel):
    text: str = Field(description="A single, standalone sentence with all pronouns resolved.")
    temporal_status: Literal["current", "historical", "uncertain"] = Field(
        default="current",
        description="'current' for present-tense facts, 'historical' for explicitly past or outdated facts, 'uncertain' if the tense is ambiguous."
    )
    valid_period: str | None = Field(
        default=None,
        description="Optional free-text period when this fact was true (e.g., 'during college', '2010-2015')."
    )

class MemoryProcessing(BaseModel):
    atomic_facts: list[AtomicFact] = Field(
        description="A list of standalone, independent sentences extracted from the text. "
        "All pronouns must be replaced with the actual entity names so each sentence makes sense in isolation."
    )
    triples: list[KnowledgeTriple]

    @field_validator("atomic_facts", mode="before")
    @classmethod
    def _coerce_strings(cls, v):
        """Accept plain strings for backward compatibility with tests and legacy code."""
        if isinstance(v, list):
            return [AtomicFact(text=x) if isinstance(x, str) else x for x in v]
        return v

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

class SupersessionDecision(BaseModel):
    outcome: Literal["A_supersedes_B", "B_supersedes_A", "contradiction", "neither"]
    explanation: str = Field(
        default="",
        description="Brief explanation of the relationship between the two facts."
    )

class ContextHint(BaseModel):
    subject: str | None = Field(
        default=None,
        description="The primary person or entity this text is mainly about, or null if unclear."
    )
    time_period: str | None = Field(
        default=None,
        description="The temporal setting (e.g., 'college years', '2010-2015'), or null if unknown or present-day."
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
        "CRITICAL: Resolve all pronouns by finding the antecedent in the full text before writing each fact. "
        "Example input: 'Alice has a cat. She loves it.' "
        "Example output: [{\"text\": \"Alice has a cat.\", \"temporal_status\": \"current\"}, "
        "{\"text\": \"Alice loves the cat.\", \"temporal_status\": \"current\"}] "
        "Never output a fact containing 'she', 'he', 'it', 'they', 'her', 'him', 'them'. "
        "Set temporal_status to 'historical' for facts stated in past tense or described as no longer true "
        "(e.g., 'She used to be a fencer', 'He was a teacher in 2010'). "
        "Set temporal_status to 'current' for present-tense or timeless facts. "
        "Set temporal_status to 'uncertain' only when the temporal state is genuinely ambiguous. "
        "Set valid_period to a short phrase when a time window is mentioned (e.g., 'during college', '2010–2015'); "
        "otherwise leave it null. "
        "IMPORTANT: If the text begins with a '[CONTEXT: ...]' line, use it only to resolve pronouns and infer "
        "temporal context — never emit it as a fact itself.\n"
        "2. Extract 'triples': Subject-predicate-object relationships from the text. "
        "Use past-tense predicates (WAS, HAD) for historical facts and present-tense (IS, HAS) for current ones."
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


def librarian_check_supersession(fact_a: str, fact_b: str) -> SupersessionDecision | None:
    """Determines whether two facts are in a supersession or contradiction relationship.

    Intended for cases where structural predicate detection (IS/WAS graph pairing) is insufficient —
    e.g., same predicate but one fact contains 'no longer' or 'used to' in the text itself.
    """
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded.")

    system_prompt = (
        "You are a fact-relationship analyzer. Given two facts about the same or similar subject, "
        "classify their relationship:\n"
        "- 'A_supersedes_B': Fact A is a more recent update, making Fact B outdated.\n"
        "- 'B_supersedes_A': Fact B is more recent, making Fact A outdated.\n"
        "- 'contradiction': The facts are directly incompatible and you cannot determine which is newer.\n"
        "- 'neither': The facts are compatible, cover different aspects, or are unrelated.\n\n"
        "Use supersession when tense or temporal language ('used to', 'no longer', 'now') implies ordering. "
        "Use 'contradiction' only for direct factual incompatibility without clear temporal ordering. "
        "When in doubt, use 'neither'."
    )

    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Fact A: "{fact_a}"\nFact B: "{fact_b}"'},
        ],
        response_format={"type": "json_object", "schema": SupersessionDecision.model_json_schema()},
        temperature=0.1,
    )

    try:
        return SupersessionDecision(**json.loads(response["choices"][0]["message"]["content"]))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Supersession check failed: {e}")
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

def extract_context_hint(text: str) -> ContextHint | None:
    """Extracts subject and temporal setting from the first chunk of a /learn input.

    The result is formatted as a [CONTEXT: ...] prefix and prepended to subsequent chunks
    so the Librarian can resolve pronouns and assign valid_period consistently across
    chunk boundaries. Works best for single-subject texts (biographies, diaries).
    """
    if librarian_llm is None:
        raise ValueError("Librarian model not loaded.")

    system_prompt = (
        "Extract a brief context summary from this text passage. "
        "Identify: (1) the primary subject — the person or entity the text is mainly about; "
        "(2) the temporal setting — a year range or life phase ('during college', '2010-2015') "
        "if one is clearly implied. Set both to null if genuinely unknown."
    )

    response = librarian_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract context from: '{text}'"},
        ],
        response_format={"type": "json_object", "schema": ContextHint.model_json_schema()},
        temperature=0.1,
    )

    try:
        return ContextHint(**json.loads(response["choices"][0]["message"]["content"]))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Context hint extraction failed: {e}")
        return None


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
