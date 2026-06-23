import json
from typing import Literal
from pydantic import BaseModel, Field, field_validator

from llm_client import get_llm_client

# --- PYDANTIC SCHEMAS ---

class Entity(BaseModel):
    name: str

class EntityExtraction(BaseModel):
    entities: list[Entity]

class KnowledgeTriple(BaseModel):
    subject: str = Field(description="The main entity (e.g., 'Hailey')")
    predicate: str = Field(description="The relationship (e.g., 'HAS', 'IS', 'MOTHER_OF')")
    object: str = Field(description="The target entity (e.g., 'Mochi')")
    supporting_fact_indices: list[int] = Field(
        default_factory=list,
        description="Zero-based indices into atomic_facts for the fact(s) that directly state this triple's information."
    )

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

class ConcurrencyDecision(BaseModel):
    outcome: Literal["concurrent", "not_concurrent"]
    explanation: str = Field(
        default="",
        description="Brief explanation of why the two facts are or are not temporally concurrent."
    )

class GroupAssignment(BaseModel):
    matching_groups: list[str] = Field(
        default_factory=list,
        description="Names of existing groups this entity belongs to. Empty list if none fit."
    )
    new_group: str | None = Field(
        default=None,
        description="Name of a new group to create, or null if an existing group covers it or no group is warranted."
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


# --- INFERENCE FUNCTIONS ---

def process_memory_chunk(text: str) -> MemoryProcessing | None:
    """Extracts atomic facts (for ChromaDB) AND triples (for the Graph) simultaneously."""
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
        "Use past-tense predicates (WAS, HAD) for historical facts and present-tense (IS, HAS) for current ones. "
        "For each triple, set supporting_fact_indices to the zero-based index (or indices) of the atomic_facts "
        "entry that directly states the information in that triple. "
        "Example: if atomic_facts[0] is 'Alice was a nurse.' and the triple is (Alice, WAS, Nurse), "
        "set supporting_fact_indices to [0]. If two facts jointly support a triple, list both indices."
    )

    print(f"[LIBRARIAN] Processing chunk for DBs: '{text[:50]}...'")
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Process this text: '{text}'"},
            ],
            schema=MemoryProcessing.model_json_schema(),
            temperature=0.1,
        )
        return MemoryProcessing(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Failed to process memory: {e}")
        return None


def extract_entities_from_text(text: str) -> EntityExtraction | None:
    """Pulls entities from user queries so we know which nodes to search in the Graph."""
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": "Extract key nouns, proper nouns, and entities from the query."},
                {"role": "user", "content": f"Extract entities from: '{text}'"},
            ],
            schema=EntityExtraction.model_json_schema(),
            temperature=0.1,
        )
        return EntityExtraction(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Entity extraction failed: {e}")
        return None


def librarian_should_merge(fact_a: str, fact_b: str) -> MergeDecision | None:
    """Returns a merge decision for two semantically similar facts."""
    system_prompt = (
        "You are a memory deduplication engine. Given two facts, decide if they are "
        "semantically equivalent or if one is a strict subset of the other. "
        "If yes, write a single merged fact that preserves the most specific information from both. "
        "If they describe genuinely different things, do NOT merge."
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Fact A: "{fact_a}"\nFact B: "{fact_b}"'},
            ],
            schema=MergeDecision.model_json_schema(),
            temperature=0.1,
        )
        return MergeDecision(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Merge decision failed: {e}")
        return None


def librarian_split_compound(fact: str) -> SplitDecision | None:
    """Returns a split decision for a potentially compound fact."""
    system_prompt = (
        "You are a memory atomization engine. A fact is 'compound' if it contains two or more "
        "independent pieces of information that would each make sense as a standalone sentence. "
        "If compound, split it into the smallest possible independent facts. "
        "Resolve all pronouns in each split so they make sense in isolation. "
        "If the statement is already a single atomic fact, set is_compound=false."
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Is this compound? "{fact}"'},
            ],
            schema=SplitDecision.model_json_schema(),
            temperature=0.1,
        )
        return SplitDecision(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Split decision failed: {e}")
        return None


def librarian_check_supersession(fact_a: str, fact_b: str) -> SupersessionDecision | None:
    """Determines whether two facts are in a supersession or contradiction relationship.

    Intended for cases where structural predicate detection (IS/WAS graph pairing) is insufficient —
    e.g., same predicate but one fact contains 'no longer' or 'used to' in the text itself.
    """
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
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Fact A: "{fact_a}"\nFact B: "{fact_b}"'},
            ],
            schema=SupersessionDecision.model_json_schema(),
            temperature=0.1,
        )
        return SupersessionDecision(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Supersession check failed: {e}")
        return None


def librarian_check_concurrency(
    fact_a: str, fact_b: str, period_a: str, period_b: str
) -> ConcurrencyDecision | None:
    """Determines whether two historical facts were true during an overlapping time window.

    Both facts must already have a non-null valid_period; the periods are passed explicitly
    so the model can use them without re-reading the fact text.
    """
    system_prompt = (
        "You are a temporal fact analyzer. Given two historical facts and their time periods, "
        "decide whether the facts were true at the same time (overlapping periods).\n"
        "- 'concurrent': The facts were both true during an overlapping period "
        "(e.g., 'college years' and '2015-2019' for the same person likely overlap).\n"
        "- 'not_concurrent': The periods are clearly distinct and non-overlapping.\n\n"
        "When in doubt, or when the periods are vague enough that you cannot rule out overlap, "
        "prefer 'not_concurrent' to avoid false positives."
    )
    user_content = (
        f'Fact A: "{fact_a}"\nTime period A: "{period_a}"\n\n'
        f'Fact B: "{fact_b}"\nTime period B: "{period_b}"'
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            schema=ConcurrencyDecision.model_json_schema(),
            temperature=0.1,
        )
        return ConcurrencyDecision(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Concurrency check failed: {e}")
        return None


def librarian_assign_groups(entity_name: str, existing_groups: list[str]) -> GroupAssignment | None:
    """Decides which thematic groups an entity belongs to.

    Returns matching existing groups and optionally a new group name when none of the existing
    groups fit. The Librarian should prefer reusing existing groups over creating new ones.
    """
    groups_list = ", ".join(f'"{g}"' for g in existing_groups) if existing_groups else "none yet"
    system_prompt = (
        "You are a memory organizer. Given an entity name and a list of existing thematic groups, "
        "decide which groups this entity belongs to. Groups are broad thematic categories like "
        "'Family', 'Friends', 'Colleagues', 'Locations', 'Organizations', 'Hobbies', 'Pets', etc.\n\n"
        "Rules:\n"
        "- Prefer matching existing groups over creating new ones.\n"
        "- Only set new_group if no existing group fits and the entity clearly warrants one.\n"
        "- Inanimate objects or abstract concepts that don't fit any category should return empty lists.\n"
        "- An entity can match multiple groups (e.g., a person who is both a friend and a colleague)."
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f'Entity: "{entity_name}"\n'
                        f"Existing groups: [{groups_list}]\n"
                        "Which groups does this entity belong to?"
                    ),
                },
            ],
            schema=GroupAssignment.model_json_schema(),
            temperature=0.1,
        )
        return GroupAssignment(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Group assignment failed: {e}")
        return None


def librarian_summarize(facts: list[str]) -> str:
    """Takes raw 'A [OWNS] B' facts and makes them a readable string for the main agent."""
    if not facts:
        return ""
    return get_llm_client().chat_text(
        messages=[
            {
                "role": "system",
                "content": "You are a concise AI. Combine these relational facts into a brief, human-readable summary.",
            },
            {"role": "user", "content": f"Summarize these facts:\n{chr(10).join(facts)}"},
        ],
        temperature=0.3,
    )


def extract_context_hint(text: str) -> ContextHint | None:
    """Extracts subject and temporal setting from the first chunk of a /learn input.

    The result is formatted as a [CONTEXT: ...] prefix and prepended to subsequent chunks
    so the Librarian can resolve pronouns and assign valid_period consistently across
    chunk boundaries. Works best for single-subject texts (biographies, diaries).
    """
    system_prompt = (
        "Extract a brief context summary from this text passage. "
        "Identify: (1) the primary subject — the person or entity the text is mainly about; "
        "(2) the temporal setting — a year range or life phase ('during college', '2010-2015') "
        "if one is clearly implied. Set both to null if genuinely unknown."
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract context from: '{text}'"},
            ],
            schema=ContextHint.model_json_schema(),
            temperature=0.1,
        )
        return ContextHint(**json.loads(output_str))
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
    """
    pass
