"""Tests for verdict engine module."""

import pytest

from src.core.config import VerdictConfig
from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    CalibratedConfidence,
    ContradictionType,
    Verdict,
    VerdictLabel,
)
from src.verdict.engine import DefaultVerdictEngine


class TestDefaultVerdictEngine:
    """Test suite for DefaultVerdictEngine."""

    def test_compute_returns_verdict(
        self, sample_claim, sample_evidence, sample_alignments
    ):
        """Compute should return Verdict object."""
        engine = DefaultVerdictEngine()

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.85,
            penalties_applied=[],
            penalty_breakdown={},
        )

        result = engine.compute(sample_claim, sample_evidence, sample_alignments, calibrated)

        assert isinstance(result, Verdict)
        assert result.claim == sample_claim

    def test_grounded_verdict_with_strong_evidence(self, sample_claim, sample_evidence):
        """Should return GROUNDED when evidence strongly supports claim."""
        config = VerdictConfig(grounded_threshold=0.7)
        engine = DefaultVerdictEngine(config)

        strong_alignment = AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[0].id,
            label=AlignmentLabel.SUPPORTS,
            confidence=0.95,
            explanation="Strong support",
            temporal_match=True,
            semantic_score=0.95,
            logical_score=0.9,
            contradiction_type=ContradictionType.NONE,
            negation_detected=False,
        )

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.9,
            penalties_applied=[],
            penalty_breakdown={},
        )

        result = engine.compute(sample_claim, sample_evidence, [strong_alignment], calibrated)

        assert result.label == VerdictLabel.GROUNDED

    def test_hallucinated_verdict_with_no_evidence(self, sample_claim):
        """Should return HALLUCINATED when no evidence found."""
        config = VerdictConfig(hallucination_threshold=0.3)
        engine = DefaultVerdictEngine(config)

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.55,
            penalties_applied=["no_evidence"],
            penalty_breakdown={"no_evidence": 0.4},
        )

        result = engine.compute(sample_claim, [], [], calibrated)

        assert result.label == VerdictLabel.HALLUCINATED
        assert result.evidence_strength == 0.0

    def test_hallucinated_verdict_with_contradiction(
        self, hallucination_claim, contradicting_evidence
    ):
        """Should return HALLUCINATED when evidence contradicts claim."""
        engine = DefaultVerdictEngine()

        contradicting_alignment = AlignmentResult(
            claim_id=hallucination_claim.id,
            evidence_id=contradicting_evidence[0].id,
            label=AlignmentLabel.CONTRADICTS,
            confidence=0.9,
            explanation="Evidence contradicts claim",
            temporal_match=True,
            semantic_score=0.85,
            logical_score=0.1,
            contradiction_type=ContradictionType.DIRECT_NEGATION,
            negation_detected=True,
        )

        calibrated = CalibratedConfidence(
            claim_id=hallucination_claim.id,
            raw_confidence=0.92,
            calibrated_confidence=0.32,
            penalties_applied=["contradiction_detected"],
            penalty_breakdown={"contradiction_detected": 0.6},
        )

        result = engine.compute(
            hallucination_claim,
            contradicting_evidence,
            [contradicting_alignment],
            calibrated,
        )

        assert result.label == VerdictLabel.HALLUCINATED
        assert result.contradiction_detected

    def test_weak_verdict_with_partial_support(self, sample_claim, sample_evidence):
        """Should return WEAK when evidence partially supports claim."""
        config = VerdictConfig(hallucination_threshold=0.3, grounded_threshold=0.7)
        engine = DefaultVerdictEngine(config)

        # WEAK_SUPPORT with high enough scores to land in WEAK range (0.3-0.7)
        # evidence_strength = 0.5 * 0.85 * (0.8 + 0.8) / 2 = 0.5 * 0.85 * 0.8 = 0.34
        weak_alignment = AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[0].id,
            label=AlignmentLabel.WEAK_SUPPORT,
            confidence=0.85,
            explanation="Partial support",
            temporal_match=True,
            semantic_score=0.8,
            logical_score=0.8,
            contradiction_type=ContradictionType.NONE,
            negation_detected=False,
        )

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.7,
            calibrated_confidence=0.55,
            penalties_applied=["weak_evidence_only"],
            penalty_breakdown={"weak_evidence_only": 0.15},
        )

        result = engine.compute(sample_claim, sample_evidence, [weak_alignment], calibrated)

        # With these scores, evidence_strength should be ~0.34, in WEAK range
        assert result.label == VerdictLabel.WEAK

    def test_hallucination_risk_calculation(self, sample_claim, sample_evidence):
        """Hallucination risk should be higher with high confidence and low evidence."""
        engine = DefaultVerdictEngine()

        # High confidence, no evidence = high risk
        calibrated_high = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.55,
            penalties_applied=["no_evidence"],
            penalty_breakdown={"no_evidence": 0.4},
        )

        result_high_risk = engine.compute(sample_claim, [], [], calibrated_high)

        # Lower confidence, strong evidence = lower risk
        strong_alignment = AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[0].id,
            label=AlignmentLabel.SUPPORTS,
            confidence=0.9,
            explanation="Strong support",
            temporal_match=True,
            semantic_score=0.9,
            logical_score=0.85,
            contradiction_type=ContradictionType.NONE,
            negation_detected=False,
        )

        calibrated_low = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.7,
            calibrated_confidence=0.75,
            penalties_applied=[],
            penalty_breakdown={},
        )

        result_low_risk = engine.compute(
            sample_claim, sample_evidence, [strong_alignment], calibrated_low
        )

        assert result_high_risk.hallucination_risk > result_low_risk.hallucination_risk

    def test_explanation_generated(self, sample_claim, sample_evidence, sample_alignments):
        """Verdict should include human-readable explanation."""
        engine = DefaultVerdictEngine()

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.85,
            penalties_applied=[],
            penalty_breakdown={},
        )

        result = engine.compute(sample_claim, sample_evidence, sample_alignments, calibrated)

        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_best_evidence_selection(self, sample_claim, sample_evidence, sample_alignments):
        """Should select best supporting evidence."""
        engine = DefaultVerdictEngine()

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.85,
            penalties_applied=[],
            penalty_breakdown={},
        )

        result = engine.compute(sample_claim, sample_evidence, sample_alignments, calibrated)

        # Should select the supporting evidence, not irrelevant
        if result.best_evidence:
            assert result.best_evidence.id == sample_evidence[0].id

    def test_risk_in_valid_range(self, sample_claim, sample_evidence, sample_alignments):
        """Hallucination risk should always be between 0 and 1."""
        engine = DefaultVerdictEngine()

        calibrated = CalibratedConfidence(
            claim_id=sample_claim.id,
            raw_confidence=0.95,
            calibrated_confidence=0.85,
            penalties_applied=[],
            penalty_breakdown={},
        )

        result = engine.compute(sample_claim, sample_evidence, sample_alignments, calibrated)

        assert 0 <= result.hallucination_risk <= 1
        assert 0 <= result.evidence_strength <= 1
