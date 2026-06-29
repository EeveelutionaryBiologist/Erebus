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

class CompoundEntityDecision(BaseModel):
    action: Literal["rewrite", "keep", "flag"]
    suggested_predicate: str | None = Field(
        default=None,
        description="UPPERCASE_SNAKE_CASE predicate to use when action is 'rewrite' (e.g. 'IS_ADVISOR_TO')."
    )
    explanation: str = Field(default="", description="Brief explanation of the decision.")

class EntityMergeDecision(BaseModel):
    should_merge: bool
    canonical_to_keep: str = Field(
        default="",
        description="The canonical entity name to keep (usually the longer, more complete form). Empty if should_merge is False."
    )
    explanation: str = Field(default="", description="Brief explanation of the decision.")

class EntityClassification(BaseModel):
    entity_type: Literal["person", "organization", "location", "concept", "role", "junk"]
    reasoning: str = Field(default="", description="Brief reasoning for the classification.")

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

def process_memory_chunk(text: str, known_entities: list[str] | None = None) -> MemoryProcessing | None:
    """Extracts atomic facts (for ChromaDB) AND triples (for the Graph) simultaneously.

    known_entities: canonical entity names already in the graph that appear in this chunk.
    When provided, the prompt instructs the model to prefer these exact names as triple
    subjects/objects rather than inventing compound role-description strings.
    """
    entity_hint = ""
    if known_entities:
        names = ", ".join(f'"{n}"' for n in known_entities)
        entity_hint = (
            f" The following entity names already exist in the knowledge graph: [{names}]. "
            "When one of these entities would naturally appear as a subject or object in a triple, "
            "use its exact name from this list — do not paraphrase or embed it in a longer string."
        )

    system_prompt = (
        "You are an advanced data extraction AI. You have two tasks:\n"
        "1. Extract 'atomic_facts': Break the text into independent, single-fact sentences. "
        "CRITICAL: Resolve all pronouns by finding the antecedent in the full text before writing each fact. "
        "Example input: 'Alice has a cat. She loves it.' "
        "Example output: [{\"text\": \"Alice has a cat.\", \"temporal_status\": \"current\"}, "
        "{\"text\": \"Alice loves the cat.\", \"temporal_status\": \"current\"}] "
        "Never output a fact containing 'she', 'he', 'it', 'they', 'her', 'him', 'them'. "
        "Set temporal_status to 'historical' for facts stated in past tense or described as no longer true "
        "(e.g., 'She used to be a fencer', 'He was a teacher in 2010', 'Alice joined X in 2013'). "
        "Any fact that contains a specific past year (e.g. 'in 2013', 'from 2008 to 2012') describing a "
        "completed event (joined, left, moved, graduated, founded) must be 'historical'. "
        "Set temporal_status to 'current' for present-tense or timeless facts. "
        "Set temporal_status to 'uncertain' only when the temporal state is genuinely ambiguous. "
        "Set valid_period to a short phrase when a time window is mentioned (e.g., 'during college', '2010–2015'); "
        "otherwise leave it null. "
        "IMPORTANT: If the text begins with a '[CONTEXT: ...]' line, use it only to resolve pronouns and infer "
        "temporal context — never emit it as a fact itself.\n"
        "2. Extract 'triples': Subject-predicate-object relationships from the text. "
        "RULE A — TRIPLE OBJECTS must be reusable named entities only. Never use as an object:\n"
        "  - Boolean strings (True, False, Yes, No) — omit the triple; the fact already captures it.\n"
        "  - Year strings (2020, 2013, etc.) — omit the triple; put the year in valid_period instead.\n"
        "  - Multi-word occupation/role strings like 'High School Art Teacher In San Francisco' "
        "— use a short generic role ('Teacher') or encode the organization as the object instead.\n"
        "RULE B — SHORT PREDICATES, max 3 words. Never pack full facts or qualifiers into predicates. "
        "BAD: FORMED_CLOSE_WORKING_FRIENDSHIP_WITH → GOOD: FRIENDS_WITH. "
        "BAD: IS_RESEARCH_ASSOCIATE_AT → GOOD: WORKED_AT. "
        "BAD: WORKED_PART_TIME_AT → GOOD: WORKED_AT. "
        "BAD: IS_CURRENTLY or IS_NOW → GOOD: IS. "
        "Never prefix a predicate with FORMERLY_ or IS_NOW_ — use WAS/IS instead.\n"
        "RULE C — PAST-TENSE PREDICATES for historical facts. "
        "If a fact's temporal_status is 'historical', its predicate must use WAS, HAD, or another "
        "past form (WORKED_AT for a past job is fine). Never use IS_CURRENTLY for a historical fact.\n"
        "If a sentence says someone holds a role AT or FOR another entity, encode the role in the predicate "
        "and use the other entity as the object. "
        "BAD: (James, IS, Advisor To Cellbridge Therapeutics). "
        "GOOD: (James, IS_ADVISOR_TO, Cellbridge Therapeutics).\n"
        "RULE D — OBJECTS MUST BE PROPER NOUNS: named persons, organizations, or locations only. "
        "Abstract noun phrases describing feelings, activities, relationships, or time periods are NOT valid objects "
        "(e.g. 'Close Working Friendship With', 'Her Love Of Community Spaces', "
        "'The Stories Her Grandmother Told Her', 'Most Of The Period From 2013 To 2017'). "
        "If no proper-noun object exists for a relationship, omit the triple — the atomic fact captures it.\n"
        "RULE E — ONE PERSON PER OBJECT: never list multiple people in a single object string "
        "(e.g. 'James And Ruth Mercer, And Tom'). Create one triple per person, or omit if not meaningful."
        + entity_hint +
        " For each triple, set supporting_fact_indices to the zero-based index (or indices) of the atomic_facts "
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


def librarian_assign_groups(
    entity_name: str,
    existing_groups: list[str],
    context_facts: list[str] | None = None,
) -> GroupAssignment | None:
    """Decides which thematic groups an entity belongs to.

    context_facts: up to 3 atomic facts mentioning this entity, so the model can see
    what kind of entity it is rather than guessing from the name alone.
    """
    groups_list = ", ".join(f'"{g}"' for g in existing_groups) if existing_groups else "none yet"
    facts_block = ""
    if context_facts:
        quoted = "\n".join(f'  - "{f}"' for f in context_facts[:3])
        facts_block = f"\nFacts about this entity:\n{quoted}"
    system_prompt = (
        "You are a memory organizer. Given an entity name and a list of existing thematic groups, "
        "decide which groups this entity belongs to.\n\n"
        "Curated group vocabulary (prefer these before creating new ones):\n"
        "Family, Friends, Colleagues, Locations, Organizations, Hobbies, Pets, Education\n\n"
        "Rules:\n"
        "- First check: if the entity is a field of study, occupation title, boolean, year, "
        "or abstract concept — return matching_groups=[] and new_group=null. "
        "Do NOT assign groups to non-person, non-organization, non-location entities.\n"
        "- Prefer matching existing groups over creating new ones.\n"
        "- Only set new_group if no existing group fits AND the entity is a specific person "
        "or named organization. new_group must be a short 1-2 word label, not a phrase. "
        "NEVER use underscores or long descriptive names for new_group.\n"
        "- An entity can match multiple groups (e.g., a person who is both a friend and a colleague).\n"
        "- Use the 'Facts about this entity' section (if provided) to determine what kind of entity it is. "
        "Do NOT let facts from the surrounding document bleed into the group assignment for this entity.\n"
        "- IMPORTANT: set new_group to null (JSON null, NOT the string 'null', 'Empty_List', '[]', or 'N/A') "
        "when no new group is needed. Never return a string where null is expected."
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f'Entity: "{entity_name}"{facts_block}\n'
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


def librarian_resolve_compound_entity(compound_name: str, contained_name: str) -> CompoundEntityDecision | None:
    """Decides whether a compound entity name encodes a predicate + contained-entity relationship.

    E.g., 'Advisor To Cellbridge Therapeutics' with contained 'Cellbridge Therapeutics'
    should be rewritten as the triple (subject, IS_ADVISOR_TO, Cellbridge Therapeutics).
    Called by Phase 5 of consolidation for each candidate compound/contained entity pair.
    """
    system_prompt = (
        "You are a knowledge graph normalizer. You have two entity names:\n"
        "1. A 'compound' name that may describe a role or relationship "
        "(e.g., 'Advisor To Cellbridge Therapeutics')\n"
        "2. A 'contained' name that appears as a substring (e.g., 'Cellbridge Therapeutics')\n\n"
        "Decide what to do:\n"
        "- 'rewrite': The compound encodes a predicate. Replace it with a proper triple "
        "(some subject, suggested_predicate, contained entity). Provide suggested_predicate as "
        "UPPERCASE_SNAKE_CASE (e.g., 'IS_ADVISOR_TO', 'WORKS_AT', 'MEMBER_OF').\n"
        "- 'keep': The compound is a legitimately distinct entity "
        "(e.g., 'Alice Kim' containing 'Alice' — two different people).\n"
        "- 'flag': Genuinely ambiguous — neither clearly a predicate nor a distinct entity.\n\n"
        "Choose 'keep' for full human names where the contained name is just a first or last name. "
        "Choose 'rewrite' when the compound prepends a role description to the contained entity. "
        "When in doubt, choose 'keep' to avoid data loss."
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f'Compound entity: "{compound_name}"\n'
                        f'Contained entity: "{contained_name}"'
                    ),
                },
            ],
            schema=CompoundEntityDecision.model_json_schema(),
            temperature=0.1,
        )
        return CompoundEntityDecision(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Compound entity resolution failed: {e}")
        return None


def librarian_should_merge_entities(
    name_a: str,
    name_b: str,
    facts_a: list[str],
    facts_b: list[str],
) -> EntityMergeDecision | None:
    """Decides whether two entity nodes refer to the same real-world entity.

    Typically called when name_a is a token-subset of name_b (e.g. 'Alice' inside
    'Alice Mercer'), which strongly suggests they are the same person written with
    different levels of specificity across chunks.
    """
    facts_a_str = "\n".join(f"  - {f}" for f in facts_a[:5]) if facts_a else "  (none)"
    facts_b_str = "\n".join(f"  - {f}" for f in facts_b[:5]) if facts_b else "  (none)"
    system_prompt = (
        "You are a knowledge graph deduplicator. Two entity nodes may refer to the same "
        "real-world person or organization. Name A appears to be a shorter form of Name B "
        "(all of A's name tokens appear in B's name).\n\n"
        "Decide:\n"
        "- should_merge=true if they refer to the same entity (e.g. 'Alice' and 'Alice Mercer' "
        "when the facts are about the same person).\n"
        "- should_merge=false if they are clearly different entities "
        "(e.g. 'Alice' as a city vs 'Alice Mercer' as a person, or different people with the same first name).\n"
        "- canonical_to_keep: the more complete name (usually Name B, the longer one).\n"
        "When in doubt and the facts do not contradict each other, prefer should_merge=true "
        "to avoid fragmentation.\n\n"
        "KEY SIGNAL — SHARED RELATIONSHIPS: If one entity's facts describe a relationship with a third party "
        "(e.g. 'Alice's partner is Jordan Kim') and the other's facts describe the same relationship with that "
        "same party ('Alice Mercer and Jordan have been together since 2016'), this is near-certain evidence they "
        "are the same entity. Return should_merge=true when you see this pattern."
    )
    # Surface any shared third-party names to make the cross-reference signal explicit.
    tokens_a = {w.lower() for f in facts_a for w in f.split() if len(w) > 3}
    tokens_b = {w.lower() for f in facts_b for w in f.split() if len(w) > 3}
    shared = tokens_a & tokens_b - {name_a.lower(), name_b.lower()}
    shared_note = (
        f"\nNote: both entities appear in facts that mention the following names/words: {', '.join(sorted(shared)[:10])}"
        if shared else ""
    )
    user_content = (
        f'Name A: "{name_a}"\nFacts about A:\n{facts_a_str}\n\n'
        f'Name B: "{name_b}"\nFacts about B:\n{facts_b_str}{shared_note}'
    )
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            schema=EntityMergeDecision.model_json_schema(),
            temperature=0.1,
        )
        return EntityMergeDecision(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Entity merge check failed: {e}")
        return None


def librarian_classify_entity(name: str, backing_facts: list[str]) -> EntityClassification | None:
    """Classifies whether an entity node is a real named entity or structural junk.

    Used by Phase 7 to flag role strings, abstract concepts, and other non-entities
    that crept into the knowledge graph as object nodes.
    """
    facts_str = "\n".join(f"  - {f}" for f in backing_facts[:5]) if backing_facts else "  (none)"
    system_prompt = (
        "You are a knowledge graph auditor. An entity node exists in the graph with the given name. "
        "Classify what kind of thing it is:\n"
        "- 'person': a named human (first+last name or title+name)\n"
        "- 'organization': a company, institution, or group\n"
        "- 'location': a place, city, region, or country\n"
        "- 'concept': an academic field, abstract idea, or named domain\n"
        "- 'role': a job title or role description (e.g. 'Chief Science Officer', 'High School Art Teacher')\n"
        "- 'junk': a boolean (True/False), year number, or meaningless fragment\n\n"
        "Use the backing facts to inform your decision."
    )
    user_content = f'Entity name: "{name}"\nBacking facts:\n{facts_str}'
    try:
        output_str = get_llm_client().chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            schema=EntityClassification.model_json_schema(),
            temperature=0.1,
        )
        return EntityClassification(**json.loads(output_str))
    except Exception as e:
        print(f"[LIBRARIAN ERROR] Entity classification failed: {e}")
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
