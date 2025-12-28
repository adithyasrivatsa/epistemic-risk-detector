"""Pytest fixtures for testing."""

import json
from typing import Any

import pytest

from src.core.config import (
    CalibrationConfig,
    Config,
    ExtractionConfig,
    RetrievalConfig,
    VerdictConfig,
)
from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    Claim,
    ClaimType,
    ContradictionType,
    EvidenceChunk,
)


class MockLLMProvider:
    """Mock LLM provider for testing without API calls."""

    def __init__(self, responses: dict[str, Any] | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self.responses.get("complete", "Mock response")

    def complete_json(self, prompt: str, schema: dict) -> dict:
        self.calls.append(prompt)
        prompt_lower = prompt.lower()

        # Default mock responses based on prompt content
        if "claim" in prompt_lower and "extract" in prompt_lower and "atomic" in prompt_lower:
            return self.responses.get(
                "extract_claims",
                {
                    "claims": [
                        {
                            "text": "Python was created in 1991",
                            "start": 0,
                            "end": 26,
                            "confidence": 0.95,
                            "is_factual": True,
                        }
                    ]
                },
            )

        # Match alignment evaluation prompts (fact-checker prompt)
        if "fact-checker" in prompt_lower or "classify the relationship" in prompt_lower:
            return self.responses.get(
                "alignment",
                {
                    "label": "SUPPORTS",
                    "confidence": 0.85,
                    "explanation": "Evidence directly supports the claim",
                    "temporal_match": True,
                    "semantic_score": 0.9,
                    "logical_score": 0.85,
                    "negation_detected": False,
                    "contradiction_type": "NONE",
                    "claim_date": None,
                    "evidence_date": None,
                },
            )

        return self.responses.get("default", {})


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Provide a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def test_config(tmp_path) -> Config:
    """Provide a test configuration with temp database."""
    return Config(
        retrieval=RetrievalConfig(
            db_path=str(tmp_path / "test_evidence.db"),
            chunk_size=256,
            chunk_overlap=32,
            top_k=3,
        ),
        calibration=CalibrationConfig(),
        verdict=VerdictConfig(),
        extraction=ExtractionConfig(max_claims=10),
    )


@pytest.fixture
def sample_claim() -> Claim:
    """Provide a sample claim for testing."""
    return Claim(
        id="test_claim_001",
        text="Python was created in 1991",
        source_span=(0, 26),
        raw_confidence=0.95,
        is_factual=True,
        claim_type=ClaimType.TEMPORAL,
        extraction_confidence=0.95,
        hedging_detected=False,
    )


@pytest.fixture
def sample_evidence() -> list[EvidenceChunk]:
    """Provide sample evidence chunks."""
    return [
        EvidenceChunk(
            id="evidence_001",
            text="Python was created by Guido van Rossum and first released in 1991.",
            source="python_facts.txt",
            similarity_score=0.92,
            chunk_index=0,
        ),
        EvidenceChunk(
            id="evidence_002",
            text="Python 3.0 was released on December 3, 2008.",
            source="python_facts.txt",
            similarity_score=0.45,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_alignments(sample_claim, sample_evidence) -> list[AlignmentResult]:
    """Provide sample alignment results."""
    return [
        AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[0].id,
            label=AlignmentLabel.SUPPORTS,
            confidence=0.9,
            explanation="Evidence directly confirms Python was released in 1991",
            temporal_match=True,
            semantic_score=0.92,
            logical_score=0.88,
            contradiction_type=ContradictionType.NONE,
            negation_detected=False,
        ),
        AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[1].id,
            label=AlignmentLabel.IRRELEVANT,
            confidence=0.7,
            explanation="Evidence about Python 3.0 is not relevant to creation date",
            temporal_match=False,
            semantic_score=0.45,
            logical_score=0.2,
            contradiction_type=ContradictionType.NONE,
            negation_detected=False,
        ),
    ]


@pytest.fixture
def hallucination_claim() -> Claim:
    """Provide a claim that should be flagged as hallucination."""
    return Claim(
        id="hallucination_001",
        text="Python 3.12 completely removed the GIL",
        source_span=(0, 38),
        raw_confidence=0.92,
        is_factual=True,
        claim_type=ClaimType.TEMPORAL,
        extraction_confidence=0.90,
        hedging_detected=False,
    )


@pytest.fixture
def contradicting_evidence() -> list[EvidenceChunk]:
    """Provide evidence that contradicts a claim."""
    return [
        EvidenceChunk(
            id="contra_001",
            text="Python 3.12 did NOT remove the GIL - it introduced per-interpreter GIL as an experimental feature.",
            source="python_facts.txt",
            similarity_score=0.88,
            chunk_index=5,
        ),
    ]
