"""Alignment evaluation between claims and evidence."""

import re
from typing import Any

from src.core.interfaces import AlignmentEvaluator
from src.core.schemas import AlignmentLabel, AlignmentResult, Claim, ContradictionType, EvidenceChunk

# Negation patterns
NEGATION_PATTERNS = [
    r"\b(?:not|no|never|none|neither|nor|nothing|nowhere|nobody)\b",
    r"\b(?:isn't|aren't|wasn't|weren't|won't|wouldn't|couldn't|shouldn't)\b",
    r"\b(?:doesn't|don't|didn't|hasn't|haven't|hadn't)\b",
    r"\b(?:cannot|can't)\b",
    r"\bNOT\b",  # Explicit caps NOT
]

# Temporal markers for date extraction
YEAR_PATTERN = r"\b(19|20)\d{2}\b"
VERSION_PATTERN = r"\bv?(\d+\.\d+(?:\.\d+)?)\b"

ALIGNMENT_PROMPT = """You are a precise fact-checker. Evaluate the relationship between a CLAIM and EVIDENCE.

CLAIM: "{claim}"

EVIDENCE: "{evidence}"

Classify the relationship as one of:
- SUPPORTS: Evidence directly confirms the claim
- WEAK_SUPPORT: Evidence partially supports but doesn't fully confirm
- CONTRADICTS: Evidence directly contradicts the claim
- IRRELEVANT: Evidence is unrelated to the claim

Also analyze:
1. Temporal alignment: Do dates/versions/timeframes match?
2. Semantic alignment: Does the meaning align?
3. Logical alignment: Is the claim logically derivable from evidence?
4. Negation: Does the evidence negate the claim?
5. Contradiction type (if CONTRADICTS):
   - DIRECT_NEGATION: "X is Y" vs "X is not Y"
   - TEMPORAL_MISMATCH: Different time periods
   - QUANTITATIVE_MISMATCH: Different numbers
   - OUTDATED_EVIDENCE: Evidence was true but is now outdated
   - PARTIAL_OVERLAP: Some parts match, some contradict

Respond with JSON:
{{
  "label": "SUPPORTS|WEAK_SUPPORT|CONTRADICTS|IRRELEVANT",
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation of why this label",
  "temporal_match": true/false,
  "semantic_score": 0.0-1.0,
  "logical_score": 0.0-1.0,
  "negation_detected": true/false,
  "contradiction_type": "NONE|DIRECT_NEGATION|TEMPORAL_MISMATCH|QUANTITATIVE_MISMATCH|OUTDATED_EVIDENCE|PARTIAL_OVERLAP",
  "claim_date": "extracted date from claim or null",
  "evidence_date": "extracted date from evidence or null"
}}"""

ALIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["SUPPORTS", "WEAK_SUPPORT", "CONTRADICTS", "IRRELEVANT"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
        "temporal_match": {"type": "boolean"},
        "semantic_score": {"type": "number", "minimum": 0, "maximum": 1},
        "logical_score": {"type": "number", "minimum": 0, "maximum": 1},
        "negation_detected": {"type": "boolean"},
        "contradiction_type": {"type": "string", "enum": ["NONE", "DIRECT_NEGATION", "TEMPORAL_MISMATCH", "QUANTITATIVE_MISMATCH", "OUTDATED_EVIDENCE", "PARTIAL_OVERLAP"]},
        "claim_date": {"type": ["string", "null"]},
        "evidence_date": {"type": ["string", "null"]},
    },
    "required": ["label", "confidence", "explanation", "temporal_match", "semantic_score", "logical_score"],
}


class LLMAlignmentEvaluator(AlignmentEvaluator):
    """Evaluates claim-evidence alignment using an LLM."""

    def __init__(self, llm_provider: Any):
        self.llm = llm_provider
        self._negation_regex = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

    def _extract_temporal_markers(self, text: str) -> list[str]:
        """Extract dates, versions, and temporal references."""
        patterns = [
            r"\b\d{4}\b",  # Years
            r"\bv?\d+\.\d+(?:\.\d+)?\b",  # Versions
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
            r"\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b",
        ]
        markers = []
        for pattern in patterns:
            markers.extend(re.findall(pattern, text, re.IGNORECASE))
        return markers

    def _extract_years(self, text: str) -> list[str]:
        """Extract year references from text."""
        return re.findall(YEAR_PATTERN, text)

    def _detect_negation(self, text: str) -> bool:
        """Detect negation words in text."""
        return bool(self._negation_regex.search(text))

    def _detect_contradiction_type(
        self, claim: str, evidence: str, claim_has_negation: bool, evidence_has_negation: bool
    ) -> ContradictionType:
        """Rule-based contradiction type detection."""
        # Direct negation: one has negation, other doesn't
        if claim_has_negation != evidence_has_negation:
            return ContradictionType.DIRECT_NEGATION

        # Temporal mismatch: different years mentioned
        claim_years = set(self._extract_years(claim))
        evidence_years = set(self._extract_years(evidence))
        if claim_years and evidence_years and not claim_years & evidence_years:
            return ContradictionType.TEMPORAL_MISMATCH

        # Quantitative mismatch: different numbers
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence))
        if claim_numbers and evidence_numbers:
            # Check if numbers are significantly different
            if not claim_numbers & evidence_numbers:
                return ContradictionType.QUANTITATIVE_MISMATCH

        return ContradictionType.PARTIAL_OVERLAP

    def _quick_temporal_check(self, claim: str, evidence: str) -> bool:
        """Quick heuristic check for temporal alignment."""
        claim_markers = set(self._extract_temporal_markers(claim))
        evidence_markers = set(self._extract_temporal_markers(evidence))

        if not claim_markers:
            return True  # No temporal claims to check

        # Check if any claim markers appear in evidence
        return bool(claim_markers & evidence_markers)

    def evaluate_single(self, claim: Claim, evidence: EvidenceChunk) -> AlignmentResult:
        """Evaluate alignment for a single claim-evidence pair."""
        prompt = ALIGNMENT_PROMPT.format(claim=claim.text, evidence=evidence.text)

        # Rule-based pre-checks
        claim_has_negation = self._detect_negation(claim.text)
        evidence_has_negation = self._detect_negation(evidence.text)

        try:
            result = self.llm.complete_json(prompt, ALIGNMENT_SCHEMA)
        except Exception as e:
            # Fallback to heuristic evaluation
            return self._heuristic_evaluate(claim, evidence, str(e))

        # Parse contradiction type
        contradiction_type_str = result.get("contradiction_type", "NONE")
        try:
            contradiction_type = ContradictionType(contradiction_type_str)
        except ValueError:
            contradiction_type = ContradictionType.NONE

        # If LLM says CONTRADICTS but didn't specify type, use rule-based detection
        if result["label"] == "CONTRADICTS" and contradiction_type == ContradictionType.NONE:
            contradiction_type = self._detect_contradiction_type(
                claim.text, evidence.text, claim_has_negation, evidence_has_negation
            )

        return AlignmentResult(
            claim_id=claim.id,
            evidence_id=evidence.id,
            label=AlignmentLabel(result["label"]),
            confidence=result["confidence"],
            explanation=result["explanation"],
            temporal_match=result["temporal_match"],
            semantic_score=result["semantic_score"],
            logical_score=result["logical_score"],
            contradiction_type=contradiction_type,
            negation_detected=result.get("negation_detected", claim_has_negation != evidence_has_negation),
            claim_date=result.get("claim_date"),
            evidence_date=result.get("evidence_date"),
        )

    def _heuristic_evaluate(
        self, claim: Claim, evidence: EvidenceChunk, error: str = ""
    ) -> AlignmentResult:
        """Fallback heuristic evaluation when LLM fails."""
        # Use similarity score as base
        semantic_score = evidence.similarity_score
        temporal_match = self._quick_temporal_check(claim.text, evidence.text)

        # Negation detection
        claim_has_negation = self._detect_negation(claim.text)
        evidence_has_negation = self._detect_negation(evidence.text)
        negation_mismatch = claim_has_negation != evidence_has_negation

        # Simple keyword overlap for logical score
        claim_words = set(claim.text.lower().split())
        evidence_words = set(evidence.text.lower().split())
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        logical_score = min(overlap * 2, 1.0)  # Scale up

        # Determine label based on scores and negation
        if negation_mismatch and semantic_score > 0.5:
            label = AlignmentLabel.CONTRADICTS
            contradiction_type = ContradictionType.DIRECT_NEGATION
        elif not temporal_match and semantic_score > 0.5:
            label = AlignmentLabel.CONTRADICTS
            contradiction_type = ContradictionType.TEMPORAL_MISMATCH
        else:
            contradiction_type = ContradictionType.NONE
            avg_score = (semantic_score + logical_score) / 2
            if avg_score > 0.7:
                label = AlignmentLabel.SUPPORTS
            elif avg_score > 0.4:
                label = AlignmentLabel.WEAK_SUPPORT
            elif avg_score < 0.2:
                label = AlignmentLabel.IRRELEVANT
            else:
                label = AlignmentLabel.WEAK_SUPPORT

        return AlignmentResult(
            claim_id=claim.id,
            evidence_id=evidence.id,
            label=label,
            confidence=0.5,  # Low confidence for heuristic
            explanation=f"Heuristic evaluation (LLM unavailable: {error[:50]})",
            temporal_match=temporal_match,
            semantic_score=semantic_score,
            logical_score=logical_score,
            contradiction_type=contradiction_type,
            negation_detected=negation_mismatch,
        )

    def evaluate(self, claim: Claim, evidence: list[EvidenceChunk]) -> list[AlignmentResult]:
        """Evaluate alignment between a claim and all evidence chunks."""
        if not evidence:
            return []

        return [self.evaluate_single(claim, e) for e in evidence]
