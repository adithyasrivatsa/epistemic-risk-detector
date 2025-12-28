"""Core module - interfaces, schemas, and configuration."""

from src.core.interfaces import (
    AlignmentEvaluator,
    ClaimExtractor,
    ConfidenceCalibrator,
    EvidenceProvider,
    LLMProvider,
    OutputRenderer,
    VerdictEngine,
)
from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    CalibratedConfidence,
    Claim,
    ClaimType,
    ContradictionType,
    EvidenceChunk,
    Verdict,
    VerdictLabel,
)

__all__ = [
    "AlignmentEvaluator",
    "AlignmentLabel",
    "AlignmentResult",
    "CalibratedConfidence",
    "Claim",
    "ClaimExtractor",
    "ClaimType",
    "ConfidenceCalibrator",
    "ContradictionType",
    "EvidenceChunk",
    "EvidenceProvider",
    "LLMProvider",
    "OutputRenderer",
    "Verdict",
    "VerdictEngine",
    "VerdictLabel",
]
