"""Strict schemas for all data structures. Schema drift breaks builds, not production."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AlignmentLabel(str, Enum):
    """Classification of relationship between claim and evidence."""

    SUPPORTS = "SUPPORTS"
    WEAK_SUPPORT = "WEAK_SUPPORT"
    CONTRADICTS = "CONTRADICTS"
    IRRELEVANT = "IRRELEVANT"


class VerdictLabel(str, Enum):
    """Final verdict for a claim."""

    GROUNDED = "GROUNDED"
    WEAK = "WEAK"
    HALLUCINATED = "HALLUCINATED"


class ClaimType(str, Enum):
    """Classification of claim structure and verifiability."""

    DIRECT = "DIRECT"  # Simple, directly verifiable: "X is Y"
    HEDGED = "HEDGED"  # Contains hedging: "might", "possibly", "believed to"
    MULTI_HOP = "MULTI_HOP"  # Requires chaining facts: "A because B and C"
    TEMPORAL = "TEMPORAL"  # Time-sensitive: "as of 2023", "recently"
    COMPARATIVE = "COMPARATIVE"  # Comparison: "faster than", "better than"
    QUANTITATIVE = "QUANTITATIVE"  # Numbers/stats: "175 billion parameters"


class Claim(BaseModel):
    """An atomic, falsifiable assertion extracted from an LLM response."""

    id: str = Field(..., description="Unique identifier for the claim")
    text: str = Field(..., min_length=1, description="The claim text")
    source_span: tuple[int, int] = Field(
        ..., description="Character offsets in original text (start, end)"
    )
    raw_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's self-reported confidence"
    )
    is_factual: bool = Field(
        True, description="Whether this is a factual claim (vs opinion)"
    )
    claim_type: ClaimType = Field(
        ClaimType.DIRECT, description="Structural classification of the claim"
    )
    extraction_confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Confidence that this was correctly extracted as atomic claim"
    )
    hedging_detected: bool = Field(
        False, description="Whether hedging language was detected"
    )
    metadata: dict = Field(default_factory=dict)

    @field_validator("source_span")
    @classmethod
    def validate_span(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("source_span start must be <= end")
        return v


class EvidenceChunk(BaseModel):
    """A chunk of evidence retrieved from the corpus."""

    id: str = Field(..., description="Unique identifier")
    text: str = Field(..., min_length=1, description="The evidence text")
    source: str = Field(..., description="Source document path or identifier")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic similarity to claim"
    )
    chunk_index: int = Field(..., ge=0, description="Position in source document")
    metadata: dict = Field(default_factory=dict)


class ContradictionType(str, Enum):
    """Types of contradictions detected."""

    NONE = "NONE"
    DIRECT_NEGATION = "DIRECT_NEGATION"  # "X is Y" vs "X is not Y"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"  # Different time periods
    QUANTITATIVE_MISMATCH = "QUANTITATIVE_MISMATCH"  # Different numbers
    OUTDATED_EVIDENCE = "OUTDATED_EVIDENCE"  # Evidence is old but was true then
    PARTIAL_OVERLAP = "PARTIAL_OVERLAP"  # Some parts match, some contradict


class AlignmentResult(BaseModel):
    """Result of evaluating alignment between a claim and evidence."""

    claim_id: str
    evidence_id: str
    label: AlignmentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Why this label was assigned")
    temporal_match: bool = Field(True, description="Whether temporal references align")
    semantic_score: float = Field(..., ge=0.0, le=1.0)
    logical_score: float = Field(..., ge=0.0, le=1.0)
    # Explicit contradiction handling
    contradiction_type: ContradictionType = Field(
        ContradictionType.NONE, description="Type of contradiction if detected"
    )
    negation_detected: bool = Field(False, description="Whether negation words flip meaning")
    evidence_date: str | None = Field(None, description="Date of evidence if extractable")
    claim_date: str | None = Field(None, description="Date referenced in claim if any")


class CalibratedConfidence(BaseModel):
    """Confidence after applying calibration penalties."""

    claim_id: str
    raw_confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    penalties_applied: list[str] = Field(default_factory=list)
    penalty_breakdown: dict[str, float] = Field(default_factory=dict)


class Verdict(BaseModel):
    """Final verdict for a claim with full explanation."""

    claim: Claim
    label: VerdictLabel
    hallucination_risk: float = Field(
        ..., ge=0.0, le=1.0, description="0=definitely grounded, 1=definitely hallucinated"
    )
    evidence_strength: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: CalibratedConfidence
    alignments: list[AlignmentResult] = Field(default_factory=list)
    best_evidence: Optional[EvidenceChunk] = None
    contradiction_detected: bool = False
    explanation: str = Field(..., description="Human-readable explanation of verdict")


class AnalysisResult(BaseModel):
    """Complete analysis result for an LLM response."""

    original_text: str
    claims: list[Claim]
    verdicts: list[Verdict]
    overall_hallucination_risk: float = Field(..., ge=0.0, le=1.0)
    summary: str
    metadata: dict = Field(default_factory=dict)
