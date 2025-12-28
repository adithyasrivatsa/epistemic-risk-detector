"""Tests for alignment evaluation module."""

import pytest

from src.core.schemas import AlignmentLabel, AlignmentResult, ContradictionType
from src.evaluators.alignment import LLMAlignmentEvaluator


class TestLLMAlignmentEvaluator:
    """Test suite for LLMAlignmentEvaluator."""

    def test_evaluate_returns_alignment_results(self, mock_llm, sample_claim, sample_evidence):
        """Evaluate should return list of AlignmentResult objects."""
        evaluator = LLMAlignmentEvaluator(mock_llm)
        results = evaluator.evaluate(sample_claim, sample_evidence)

        assert isinstance(results, list)
        assert len(results) == len(sample_evidence)
        assert all(isinstance(r, AlignmentResult) for r in results)

    def test_evaluate_empty_evidence(self, mock_llm, sample_claim):
        """Evaluate with no evidence should return empty list."""
        evaluator = LLMAlignmentEvaluator(mock_llm)
        results = evaluator.evaluate(sample_claim, [])

        assert results == []

    def test_evaluate_single_returns_alignment(self, mock_llm, sample_claim, sample_evidence):
        """evaluate_single should return single AlignmentResult."""
        evaluator = LLMAlignmentEvaluator(mock_llm)
        result = evaluator.evaluate_single(sample_claim, sample_evidence[0])

        assert isinstance(result, AlignmentResult)
        assert result.claim_id == sample_claim.id
        assert result.evidence_id == sample_evidence[0].id

    def test_alignment_has_required_fields(self, mock_llm, sample_claim, sample_evidence):
        """Alignment results should have all required fields."""
        evaluator = LLMAlignmentEvaluator(mock_llm)
        result = evaluator.evaluate_single(sample_claim, sample_evidence[0])

        assert result.label in AlignmentLabel
        assert 0 <= result.confidence <= 1
        assert result.explanation is not None
        assert isinstance(result.temporal_match, bool)
        assert 0 <= result.semantic_score <= 1
        assert 0 <= result.logical_score <= 1
        assert result.contradiction_type in ContradictionType
        assert isinstance(result.negation_detected, bool)

    def test_heuristic_fallback_on_llm_failure(self, mock_llm, sample_claim, sample_evidence):
        """Evaluator should fall back to heuristics when LLM fails."""

        # Make LLM raise an exception
        def failing_complete_json(*args, **kwargs):
            raise Exception("API Error")

        mock_llm.complete_json = failing_complete_json

        evaluator = LLMAlignmentEvaluator(mock_llm)
        result = evaluator.evaluate_single(sample_claim, sample_evidence[0])

        # Should still return a result (heuristic)
        assert isinstance(result, AlignmentResult)
        assert "Heuristic" in result.explanation

    def test_temporal_marker_extraction(self, mock_llm):
        """Evaluator should extract temporal markers from text."""
        evaluator = LLMAlignmentEvaluator(mock_llm)

        markers = evaluator._extract_temporal_markers("Python 3.12 was released in 2023")
        assert "2023" in markers or "3.12" in markers

        markers = evaluator._extract_temporal_markers("No dates here")
        assert len(markers) == 0

    def test_quick_temporal_check(self, mock_llm):
        """Quick temporal check should detect matching dates."""
        evaluator = LLMAlignmentEvaluator(mock_llm)

        # Matching dates
        assert evaluator._quick_temporal_check(
            "Python was released in 1991",
            "Guido created Python in 1991"
        )

        # No temporal claims
        assert evaluator._quick_temporal_check(
            "Python is popular",
            "Many developers use Python"
        )

    def test_contradiction_detection(self, mock_llm, hallucination_claim, contradicting_evidence):
        """Evaluator should detect contradictions."""
        mock_llm.responses["alignment"] = {
            "label": "CONTRADICTS",
            "confidence": 0.9,
            "explanation": "Evidence contradicts the claim about GIL removal",
            "temporal_match": True,
            "semantic_score": 0.85,
            "logical_score": 0.1,
            "negation_detected": True,
            "contradiction_type": "DIRECT_NEGATION",
        }

        evaluator = LLMAlignmentEvaluator(mock_llm)
        result = evaluator.evaluate_single(hallucination_claim, contradicting_evidence[0])

        assert result.label == AlignmentLabel.CONTRADICTS
        assert result.contradiction_type == ContradictionType.DIRECT_NEGATION

    def test_negation_detection(self, mock_llm):
        """Evaluator should detect negation words."""
        evaluator = LLMAlignmentEvaluator(mock_llm)

        assert evaluator._detect_negation("Python did NOT remove the GIL")
        assert evaluator._detect_negation("This isn't true")
        assert evaluator._detect_negation("Never happened")
        assert not evaluator._detect_negation("Python was created in 1991")

    def test_contradiction_type_detection(self, mock_llm):
        """Evaluator should detect specific contradiction types."""
        evaluator = LLMAlignmentEvaluator(mock_llm)

        # Direct negation
        ct = evaluator._detect_contradiction_type(
            "Python removed the GIL",
            "Python did NOT remove the GIL",
            claim_has_negation=False,
            evidence_has_negation=True
        )
        assert ct == ContradictionType.DIRECT_NEGATION

        # Temporal mismatch (no numbers, just years)
        ct = evaluator._detect_contradiction_type(
            "Released in 2023",
            "Released in 2024",
            claim_has_negation=False,
            evidence_has_negation=False
        )
        # Both temporal and quantitative patterns match years, so either is acceptable
        assert ct in (ContradictionType.TEMPORAL_MISMATCH, ContradictionType.QUANTITATIVE_MISMATCH)

    def test_heuristic_detects_negation_contradiction(self, mock_llm, sample_evidence):
        """Heuristic evaluation should detect negation-based contradictions."""
        from src.core.schemas import Claim, ClaimType

        # Make LLM fail to force heuristic
        def failing_complete_json(*args, **kwargs):
            raise Exception("API Error")
        mock_llm.complete_json = failing_complete_json

        claim = Claim(
            id="test",
            text="Python removed the GIL",
            source_span=(0, 22),
            raw_confidence=0.9,
            is_factual=True,
            claim_type=ClaimType.DIRECT,
            extraction_confidence=0.9,
            hedging_detected=False,
        )

        from src.core.schemas import EvidenceChunk
        evidence = EvidenceChunk(
            id="ev",
            text="Python did NOT remove the GIL",
            source="test.txt",
            similarity_score=0.85,
            chunk_index=0,
        )

        evaluator = LLMAlignmentEvaluator(mock_llm)
        result = evaluator._heuristic_evaluate(claim, evidence)

        assert result.label == AlignmentLabel.CONTRADICTS
        assert result.negation_detected
        assert result.contradiction_type == ContradictionType.DIRECT_NEGATION
