"""Tests for confidence calibration module."""

import pytest

from src.calibrators.confidence import PenaltyBasedCalibrator
from src.core.config import CalibrationConfig
from src.core.schemas import AlignmentLabel, AlignmentResult, CalibratedConfidence, Claim


class TestPenaltyBasedCalibrator:
    """Test suite for PenaltyBasedCalibrator."""

    def test_calibrate_returns_calibrated_confidence(
        self, sample_claim, sample_alignments, sample_evidence
    ):
        """Calibrate should return CalibratedConfidence object."""
        calibrator = PenaltyBasedCalibrator()
        result = calibrator.calibrate(sample_claim, sample_alignments, sample_evidence)

        assert isinstance(result, CalibratedConfidence)
        assert result.claim_id == sample_claim.id

    def test_no_evidence_penalty(self, sample_claim):
        """Should apply penalty when no evidence found."""
        config = CalibrationConfig(no_evidence_penalty=0.4)
        calibrator = PenaltyBasedCalibrator(config)

        result = calibrator.calibrate(sample_claim, [], [])

        assert "no_evidence" in result.penalties_applied
        assert result.calibrated_confidence < result.raw_confidence
        assert result.penalty_breakdown["no_evidence"] == 0.4

    def test_contradiction_penalty(self, sample_claim, sample_evidence):
        """Should apply penalty when contradiction detected."""
        config = CalibrationConfig(contradiction_penalty=0.6)
        calibrator = PenaltyBasedCalibrator(config)

        # Create contradicting alignment
        contradicting_alignment = AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[0].id,
            label=AlignmentLabel.CONTRADICTS,
            confidence=0.9,
            explanation="Evidence contradicts claim",
            temporal_match=True,
            semantic_score=0.8,
            logical_score=0.1,
        )

        result = calibrator.calibrate(sample_claim, [contradicting_alignment], sample_evidence)

        assert "contradiction_detected" in result.penalties_applied
        assert result.calibrated_confidence < result.raw_confidence

    def test_vague_language_penalty(self, sample_evidence, sample_alignments):
        """Should apply penalty for vague language in claim."""
        config = CalibrationConfig(vague_language_penalty=0.2)
        calibrator = PenaltyBasedCalibrator(config)

        vague_claim = Claim(
            id="vague_001",
            text="Python might have been created around 1991",
            source_span=(0, 42),
            raw_confidence=0.9,
            is_factual=True,
        )

        result = calibrator.calibrate(vague_claim, sample_alignments, sample_evidence)

        assert "vague_language" in result.penalties_applied

    def test_calibrated_confidence_in_range(
        self, sample_claim, sample_alignments, sample_evidence
    ):
        """Calibrated confidence should always be between 0 and 1."""
        calibrator = PenaltyBasedCalibrator()

        # Test with various penalty combinations
        result = calibrator.calibrate(sample_claim, sample_alignments, sample_evidence)
        assert 0 <= result.calibrated_confidence <= 1

        # Test with no evidence (heavy penalty)
        result = calibrator.calibrate(sample_claim, [], [])
        assert 0 <= result.calibrated_confidence <= 1

    def test_strong_evidence_boost(self, sample_claim, sample_evidence):
        """Should boost confidence slightly with strong supporting evidence."""
        calibrator = PenaltyBasedCalibrator()

        # Create strong supporting alignment
        strong_alignment = AlignmentResult(
            claim_id=sample_claim.id,
            evidence_id=sample_evidence[0].id,
            label=AlignmentLabel.SUPPORTS,
            confidence=0.95,
            explanation="Strong support",
            temporal_match=True,
            semantic_score=0.95,
            logical_score=0.9,
        )

        result = calibrator.calibrate(sample_claim, [strong_alignment], sample_evidence)

        # With strong evidence and no penalties, might get a small boost
        # or at least no significant reduction
        assert result.calibrated_confidence >= result.raw_confidence - 0.1

    def test_vague_pattern_detection(self):
        """Should detect various vague language patterns."""
        calibrator = PenaltyBasedCalibrator()

        vague_texts = [
            "Python might be popular",
            "It could have been released in 1991",
            "Perhaps this is true",
            "Some developers use Python",
            "It seems to work",
            "Around 1991",
            "I think Python is good",
        ]

        for text in vague_texts:
            assert calibrator._detect_vague_language(text), f"Should detect vague: {text}"

        non_vague_texts = [
            "Python was released in 1991",
            "The GIL prevents concurrent execution",
            "Django is a web framework",
        ]

        for text in non_vague_texts:
            assert not calibrator._detect_vague_language(text), f"Should not detect vague: {text}"

    def test_penalty_breakdown_accuracy(self, sample_claim):
        """Penalty breakdown should accurately reflect applied penalties."""
        config = CalibrationConfig(
            no_evidence_penalty=0.4,
            vague_language_penalty=0.2,
        )
        calibrator = PenaltyBasedCalibrator(config)

        vague_claim = Claim(
            id="vague_001",
            text="Python might have been created",
            source_span=(0, 30),
            raw_confidence=0.9,
            is_factual=True,
        )

        result = calibrator.calibrate(vague_claim, [], [])

        # Should have both penalties
        total_penalty = sum(result.penalty_breakdown.values())
        expected_reduction = result.raw_confidence - result.calibrated_confidence

        # Allow small tolerance for floating point
        assert abs(total_penalty - expected_reduction) < 0.01
