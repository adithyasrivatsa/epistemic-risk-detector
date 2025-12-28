"""Tests for claim extraction module."""

import pytest

from src.core.config import ExtractionConfig
from src.core.schemas import Claim, ClaimType
from src.extractors.claim_extractor import LLMClaimExtractor


class TestLLMClaimExtractor:
    """Test suite for LLMClaimExtractor."""

    def test_extract_returns_claims(self, mock_llm):
        """Extractor should return list of Claim objects."""
        extractor = LLMClaimExtractor(mock_llm)
        claims = extractor.extract("Python was created in 1991.")

        assert isinstance(claims, list)
        assert len(claims) > 0
        assert all(isinstance(c, Claim) for c in claims)

    def test_extract_empty_text(self, mock_llm):
        """Extractor should handle empty input gracefully."""
        extractor = LLMClaimExtractor(mock_llm)
        claims = extractor.extract("")

        assert claims == []

    def test_extract_with_confidence_returns_metadata(self, mock_llm):
        """extract_with_confidence should return claims and metadata."""
        extractor = LLMClaimExtractor(mock_llm)
        claims, metadata = extractor.extract_with_confidence("Python was created in 1991.")

        assert isinstance(claims, list)
        assert isinstance(metadata, dict)
        assert "total_extracted" in metadata
        assert "claim_types" in metadata

    def test_claim_has_required_fields(self, mock_llm):
        """Extracted claims should have all required fields."""
        extractor = LLMClaimExtractor(mock_llm)
        claims = extractor.extract("Python was created in 1991.")

        if claims:
            claim = claims[0]
            assert claim.id is not None
            assert claim.text is not None
            assert claim.source_span is not None
            assert 0 <= claim.raw_confidence <= 1
            assert claim.claim_type in ClaimType
            assert 0 <= claim.extraction_confidence <= 1
            assert isinstance(claim.hedging_detected, bool)

    def test_respects_max_claims_config(self, mock_llm):
        """Extractor should respect max_claims configuration."""
        mock_llm.responses["extract_claims"] = {
            "claims": [
                {"text": f"Claim {i}", "start": i * 10, "end": i * 10 + 8, "confidence": 0.9, "is_factual": True}
                for i in range(20)
            ]
        }

        config = ExtractionConfig(max_claims=5)
        extractor = LLMClaimExtractor(mock_llm, config)
        claims = extractor.extract("Many claims here...")

        assert len(claims) <= 5

    def test_filters_opinions_when_configured(self, mock_llm):
        """Extractor should filter opinions when include_opinions=False."""
        mock_llm.responses["extract_claims"] = {
            "claims": [
                {"text": "Python is great", "start": 0, "end": 15, "confidence": 0.9, "is_factual": False},
                {"text": "Python was created in 1991", "start": 17, "end": 43, "confidence": 0.95, "is_factual": True},
            ]
        }

        config = ExtractionConfig(include_opinions=False)
        extractor = LLMClaimExtractor(mock_llm, config)
        claims = extractor.extract("Python is great. Python was created in 1991.")

        # Should only include factual claims
        assert all(c.is_factual for c in claims)

    def test_generates_deterministic_ids(self, mock_llm):
        """Same claim text and position should generate same ID."""
        extractor = LLMClaimExtractor(mock_llm)

        id1 = extractor._generate_claim_id("Python was created in 1991", 0)
        id2 = extractor._generate_claim_id("Python was created in 1991", 0)
        id3 = extractor._generate_claim_id("Python was created in 1991", 10)

        assert id1 == id2
        assert id1 != id3  # Different position = different ID

    def test_validates_spans(self, mock_llm):
        """Extractor should validate and fix claim spans."""
        mock_llm.responses["extract_claims"] = {
            "claims": [
                {"text": "Python", "start": 100, "end": 50, "confidence": 0.9, "is_factual": True}  # Invalid span
            ]
        }

        extractor = LLMClaimExtractor(mock_llm)
        # Should not raise, should fix or skip invalid spans
        claims = extractor.extract("Python was created in 1991.")

        for claim in claims:
            assert claim.source_span[0] <= claim.source_span[1]

    def test_detects_hedging_language(self, mock_llm):
        """Extractor should detect hedging language in claims."""
        extractor = LLMClaimExtractor(mock_llm)

        # Test hedging detection
        assert extractor._detect_hedging("Python might be popular")
        assert extractor._detect_hedging("It is believed that Python is fast")
        assert extractor._detect_hedging("Python could possibly be the best")
        assert not extractor._detect_hedging("Python was created in 1991")

    def test_detects_claim_types(self, mock_llm):
        """Extractor should correctly classify claim types."""
        extractor = LLMClaimExtractor(mock_llm)

        assert extractor._detect_claim_type("Python might be popular") == ClaimType.HEDGED
        assert extractor._detect_claim_type("Python is faster than Java") == ClaimType.COMPARATIVE
        assert extractor._detect_claim_type("GPT-3 has 175 billion parameters") == ClaimType.QUANTITATIVE
        assert extractor._detect_claim_type("As of 2023, Python is popular") == ClaimType.TEMPORAL
        assert extractor._detect_claim_type("X because Y and Z") == ClaimType.MULTI_HOP
        assert extractor._detect_claim_type("Python is a language") == ClaimType.DIRECT

    def test_hedged_claims_flagged(self, mock_llm):
        """Claims with hedging should be flagged."""
        mock_llm.responses["extract_claims"] = {
            "claims": [
                {"text": "Python might be the fastest language", "start": 0, "end": 36, "confidence": 0.7, "is_factual": True}
            ]
        }

        extractor = LLMClaimExtractor(mock_llm)
        claims = extractor.extract("Python might be the fastest language")

        if claims:
            assert claims[0].hedging_detected
            assert claims[0].claim_type == ClaimType.HEDGED
