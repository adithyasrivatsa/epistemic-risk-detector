"""Confidence calibration with evidence-based penalties."""

import re

from src.core.config import CalibrationConfig
from src.core.interfaces import ConfidenceCalibrator
from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    CalibratedConfidence,
    Claim,
    EvidenceChunk,
)

# Patterns indicating vague or hedging language
VAGUE_PATTERNS = [
    r"\b(?:might|may|could|possibly|perhaps|probably|likely|unlikely)\b",
    r"\b(?:some|many|few|several|various|certain)\b",
    r"\b(?:often|sometimes|occasionally|rarely|usually|generally)\b",
    r"\b(?:seems?|appears?|suggests?)\b",
    r"\b(?:around|approximately|about|roughly)\b",
    r"\b(?:I think|I believe|in my opinion)\b",
]


class PenaltyBasedCalibrator(ConfidenceCalibrator):
    """Calibrates confidence by applying evidence-based penalties."""

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self._vague_regex = re.compile("|".join(VAGUE_PATTERNS), re.IGNORECASE)

    def _detect_vague_language(self, text: str) -> bool:
        """Check if text contains vague or hedging language."""
        return bool(self._vague_regex.search(text))

    def _has_contradiction(self, alignments: list[AlignmentResult]) -> bool:
        """Check if any alignment indicates contradiction."""
        return any(a.label == AlignmentLabel.CONTRADICTS for a in alignments)

    def _has_strong_support(self, alignments: list[AlignmentResult]) -> bool:
        """Check if any alignment provides strong support."""
        return any(
            a.label == AlignmentLabel.SUPPORTS and a.confidence > 0.7 for a in alignments
        )

    def _compute_evidence_quality(
        self, alignments: list[AlignmentResult], evidence: list[EvidenceChunk]
    ) -> float:
        """Compute overall evidence quality score."""
        if not alignments:
            return 0.0

        # Weight by alignment label
        label_weights = {
            AlignmentLabel.SUPPORTS: 1.0,
            AlignmentLabel.WEAK_SUPPORT: 0.5,
            AlignmentLabel.CONTRADICTS: -0.5,
            AlignmentLabel.IRRELEVANT: 0.0,
        }

        scores = []
        for alignment in alignments:
            weight = label_weights[alignment.label]
            # Factor in similarity score from evidence
            evidence_chunk = next(
                (e for e in evidence if e.id == alignment.evidence_id), None
            )
            sim_score = evidence_chunk.similarity_score if evidence_chunk else 0.5
            scores.append(weight * alignment.confidence * sim_score)

        return max(0.0, min(1.0, sum(scores) / len(scores) + 0.5))

    def calibrate(
        self,
        claim: Claim,
        alignments: list[AlignmentResult],
        evidence: list[EvidenceChunk],
    ) -> CalibratedConfidence:
        """Calibrate confidence with penalties."""
        raw = claim.raw_confidence
        penalties: list[str] = []
        penalty_breakdown: dict[str, float] = {}

        calibrated = raw

        # Penalty 1: No evidence found
        if not evidence:
            penalty = self.config.no_evidence_penalty
            calibrated -= penalty
            penalties.append("no_evidence")
            penalty_breakdown["no_evidence"] = penalty

        # Penalty 2: Contradiction detected
        elif self._has_contradiction(alignments):
            penalty = self.config.contradiction_penalty
            calibrated -= penalty
            penalties.append("contradiction_detected")
            penalty_breakdown["contradiction_detected"] = penalty

        # Penalty 3: Only weak evidence
        elif not self._has_strong_support(alignments):
            penalty = self.config.weak_evidence_penalty
            calibrated -= penalty
            penalties.append("weak_evidence_only")
            penalty_breakdown["weak_evidence_only"] = penalty

        # Penalty 4: Vague language in claim
        if self._detect_vague_language(claim.text):
            penalty = self.config.vague_language_penalty
            calibrated -= penalty
            penalties.append("vague_language")
            penalty_breakdown["vague_language"] = penalty

        # Bonus: Strong supporting evidence can boost confidence slightly
        if self._has_strong_support(alignments) and not self._has_contradiction(alignments):
            evidence_quality = self._compute_evidence_quality(alignments, evidence)
            if evidence_quality > 0.7:
                boost = min(0.1, (evidence_quality - 0.7) * 0.5)
                calibrated += boost
                if boost > 0:
                    penalties.append("strong_evidence_boost")
                    penalty_breakdown["strong_evidence_boost"] = -boost

        # Clamp to valid range
        calibrated = max(0.0, min(1.0, calibrated))

        return CalibratedConfidence(
            claim_id=claim.id,
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            penalties_applied=penalties,
            penalty_breakdown=penalty_breakdown,
        )
