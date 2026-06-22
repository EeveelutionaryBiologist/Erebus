"""
Tests for librarian.py LLM inference functions.

All tests here are marked `requires_model` because they need:
  - Embedding/qwen2.5-3b-instruct-q4_k_m.gguf present on disk
  - Several GB of RAM for the model to load

Run with:
  pytest -m requires_model

Skip with:
  pytest -m "not requires_model"   (default CI / dev behaviour)
"""

import pytest


@pytest.fixture(scope="module")
def librarian():
    """Load the LLM backend once for the whole module (slow ~30s startup)."""
    from llm_client import load_llm_client
    load_llm_client()
    import librarian as lib
    return lib


@pytest.mark.requires_model
@pytest.mark.slow
class TestProcessMemoryChunk:
    def test_returns_memory_processing_object(self, librarian):
        result = librarian.process_memory_chunk("Alice owns a bakery in Paris.")
        assert result is not None
        assert isinstance(result.atomic_facts, list)
        assert isinstance(result.triples, list)

    def test_atomic_facts_are_non_empty_strings(self, librarian):
        result = librarian.process_memory_chunk("Bob likes pizza and hates broccoli.")
        assert result is not None
        assert all(isinstance(f, str) and f for f in [fact.text for fact in result.atomic_facts])

    def test_triples_have_subject_predicate_object(self, librarian):
        result = librarian.process_memory_chunk("Carol works at Acme Corp.")
        assert result is not None
        for triple in result.triples:
            assert triple.subject and triple.predicate and triple.object

    def test_pronoun_resolution(self, librarian):
        """Atomic facts must not contain bare pronouns like 'she' or 'he'."""
        result = librarian.process_memory_chunk("Hailey has a cat. She loves it.")
        assert result is not None
        combined = " ".join([fact.text for fact in result.atomic_facts]).lower()

        assert "she " not in combined and " he " not in combined


@pytest.mark.requires_model
@pytest.mark.slow
class TestExtractEntitiesFromText:
    def test_returns_entity_extraction_object(self, librarian):
        result = librarian.extract_entities_from_text("What does Alice think about Paris?")
        assert result is not None
        assert isinstance(result.entities, list)

    def test_known_entity_is_extracted(self, librarian):
        result = librarian.extract_entities_from_text("Tell me about Alice.")
        assert result is not None
        names = [e.name for e in result.entities]
        assert any("alice" in n.lower() for n in names)


@pytest.mark.requires_model
@pytest.mark.slow
class TestLibrarianShouldMerge:
    def test_identical_facts_should_merge(self, librarian):
        result = librarian.librarian_should_merge(
            "Alice owns a bakery.",
            "Alice has a bakery.",
        )
        assert result is not None
        assert result.should_merge is True
        assert result.merged_fact

    def test_unrelated_facts_should_not_merge(self, librarian):
        result = librarian.librarian_should_merge(
            "Alice owns a bakery.",
            "The sky is blue.",
        )
        assert result is not None
        assert result.should_merge is False


@pytest.mark.requires_model
@pytest.mark.slow
class TestLibrarianSplitCompound:
    def test_atomic_fact_not_split(self, librarian):
        result = librarian.librarian_split_compound("Alice owns a bakery.")
        assert result is not None
        assert result.is_compound is False

    def test_compound_fact_is_split(self, librarian):
        result = librarian.librarian_split_compound(
            "Alice owns a bakery in Paris and she also runs a café in Lyon."
        )
        assert result is not None
        assert result.is_compound is True
        assert len(result.split_facts) >= 2
