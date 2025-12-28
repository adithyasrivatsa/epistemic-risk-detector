"""Tests for output rendering modules."""

import json

import pytest

from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    AnalysisResult,
    CalibratedConfidence,
    Claim,
    EvidenceChunk,
    Verdict,
    VerdictLabel,
)
from src.renderers.cli import CLIRenderer
from src.renderers.structured import StructuredRenderer


@pytest.fixture
def sample_analysis_result():
    """Create a sample analysis result for testing."""
    claim = Claim(
        id="test_001",
        text="Python was created in 1991",
        source_span=(0, 26),
        raw_confidence=0.95,
        is_factual=True,
    )

    evidence = EvidenceChunk(
        id="evidence_001",
        text="Python was created by Guido van Rossum in 1991.",
        source="python_facts.txt",
        similarity_score=0.92,
        chunk_index=0,
    )

    alignment = AlignmentResult(
        claim_id=claim.id,
        evidence_id=evidence.id,
        label=AlignmentLabel.SUPPORTS,
        confidence=0.9,
        explanation="Evidence directly confirms the claim",
        temporal_match=True,
        semantic_score=0.92,
        logical_score=0.88,
    )

    calibrated = CalibratedConfidence(
        claim_id=claim.id,
        raw_confidence=0.95,
        calibrated_confidence=0.9,
        penalties_applied=[],
        penalty_breakdown={},
    )

    verdict = Verdict(
        claim=claim,
        label=VerdictLabel.GROUNDED,
        hallucination_risk=0.15,
        evidence_strength=0.85,
        calibrated_confidence=calibrated,
        alignments=[alignment],
        best_evidence=evidence,
        contradiction_detected=False,
        explanation="Strong evidence supports this claim.",
    )

    return AnalysisResult(
        original_text="Python was created in 1991.",
        claims=[claim],
        verdicts=[verdict],
        overall_hallucination_risk=0.15,
        summary="All claims appear grounded.",
        metadata={},
    )


class TestCLIRenderer:
    """Test suite for CLIRenderer."""

    def test_render_returns_string(self, sample_analysis_result):
        """Render should return a string."""
        renderer = CLIRenderer()
        output = renderer.render(sample_analysis_result)

        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_verdict_returns_string(self, sample_analysis_result):
        """render_verdict should return a string."""
        renderer = CLIRenderer()
        output = renderer.render_verdict(sample_analysis_result.verdicts[0])

        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_includes_claim_text(self, sample_analysis_result):
        """Output should include the claim text."""
        renderer = CLIRenderer()
        output = renderer.render(sample_analysis_result)

        assert "Python" in output

    def test_render_includes_verdict_label(self, sample_analysis_result):
        """Output should include the verdict label."""
        renderer = CLIRenderer()
        output = renderer.render(sample_analysis_result)

        assert "GROUNDED" in output

    def test_render_includes_risk_score(self, sample_analysis_result):
        """Output should include risk information."""
        renderer = CLIRenderer()
        output = renderer.render(sample_analysis_result)

        # Should have some indication of risk
        assert "Risk" in output or "risk" in output

    def test_render_handles_empty_results(self):
        """Should handle analysis with no claims."""
        renderer = CLIRenderer()

        empty_result = AnalysisResult(
            original_text="No factual claims here.",
            claims=[],
            verdicts=[],
            overall_hallucination_risk=0.0,
            summary="No factual claims found.",
            metadata={},
        )

        output = renderer.render(empty_result)
        assert isinstance(output, str)


class TestStructuredRenderer:
    """Test suite for StructuredRenderer."""

    def test_render_returns_valid_json(self, sample_analysis_result):
        """Render should return valid JSON string."""
        renderer = StructuredRenderer()
        output = renderer.render(sample_analysis_result)

        # Should be valid JSON
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_render_verdict_returns_valid_json(self, sample_analysis_result):
        """render_verdict should return valid JSON string."""
        renderer = StructuredRenderer()
        output = renderer.render_verdict(sample_analysis_result.verdicts[0])

        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_render_includes_required_fields(self, sample_analysis_result):
        """JSON output should include all required fields."""
        renderer = StructuredRenderer()
        output = renderer.render(sample_analysis_result)
        parsed = json.loads(output)

        assert "original_text" in parsed
        assert "claims_count" in parsed
        assert "overall_hallucination_risk" in parsed
        assert "verdicts" in parsed
        assert "statistics" in parsed

    def test_verdict_structure(self, sample_analysis_result):
        """Verdict JSON should have correct structure."""
        renderer = StructuredRenderer()
        output = renderer.render(sample_analysis_result)
        parsed = json.loads(output)

        verdict = parsed["verdicts"][0]
        assert "claim" in verdict
        assert "verdict" in verdict
        assert "hallucination_risk" in verdict
        assert "evidence_strength" in verdict
        assert "confidence" in verdict
        assert "explanation" in verdict

    def test_render_for_web_returns_dict(self, sample_analysis_result):
        """render_for_web should return dictionary directly."""
        renderer = StructuredRenderer()
        output = renderer.render_for_web(sample_analysis_result)

        assert isinstance(output, dict)
        assert "highlighted_claims" in output

    def test_highlighted_claims_structure(self, sample_analysis_result):
        """Highlighted claims should have span information."""
        renderer = StructuredRenderer()
        output = renderer.render_for_web(sample_analysis_result)

        highlighted = output["highlighted_claims"]
        assert len(highlighted) > 0

        for item in highlighted:
            assert "start" in item
            assert "end" in item
            assert "verdict" in item
            assert "claim_id" in item

    def test_statistics_accuracy(self, sample_analysis_result):
        """Statistics should accurately count verdicts."""
        renderer = StructuredRenderer()
        output = renderer.render(sample_analysis_result)
        parsed = json.loads(output)

        stats = parsed["statistics"]
        assert stats["grounded"] == 1
        assert stats["weak"] == 0
        assert stats["hallucinated"] == 0

    def test_pretty_vs_compact_output(self, sample_analysis_result):
        """Pretty output should be longer than compact."""
        pretty_renderer = StructuredRenderer(pretty=True)
        compact_renderer = StructuredRenderer(pretty=False)

        pretty_output = pretty_renderer.render(sample_analysis_result)
        compact_output = compact_renderer.render(sample_analysis_result)

        assert len(pretty_output) > len(compact_output)

        # Both should be valid JSON
        json.loads(pretty_output)
        json.loads(compact_output)
