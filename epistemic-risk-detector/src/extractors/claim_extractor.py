"""Claim extraction from LLM responses using deterministic prompting."""

import hashlib
import re
from typing import Any

from src.core.config import ExtractionConfig
from src.core.interfaces import ClaimExtractor
from src.core.schemas import Claim, ClaimType

# Hedging patterns that indicate uncertainty
HEDGING_PATTERNS = [
    r"\b(?:might|may|could|possibly|perhaps|probably|likely|unlikely)\b",
    r"\b(?:it is believed|it is thought|some say|reportedly|allegedly)\b",
    r"\b(?:seems?|appears?|suggests?|indicates?)\b",
    r"\b(?:I think|I believe|in my opinion|arguably)\b",
    r"\b(?:generally|typically|usually|often|sometimes)\b",
]

# Multi-hop indicators
MULTI_HOP_PATTERNS = [
    r"\b(?:because|therefore|thus|hence|consequently|as a result)\b",
    r"\b(?:since|given that|due to|owing to)\b",
    r"\b(?:which means|this implies|leading to)\b",
]

# Temporal patterns
TEMPORAL_PATTERNS = [
    r"\b(?:as of|since|until|before|after|recently|currently|now)\b",
    r"\b(?:in \d{4}|during \d{4}|by \d{4})\b",
    r"\b(?:last year|this year|next year)\b",
]

# Comparative patterns
COMPARATIVE_PATTERNS = [
    r"\b(?:faster|slower|better|worse|more|less|larger|smaller)\s+than\b",
    r"\b(?:compared to|relative to|versus|vs\.?)\b",
    r"\b(?:the most|the least|the best|the worst)\b",
]

# Quantitative patterns
QUANTITATIVE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:billion|million|thousand|percent|%)\b",
    r"\b(?:approximately|about|around|roughly)\s*\d+\b",
]

EXTRACTION_PROMPT = """You are a precise claim extractor. Your task is to decompose the following text into atomic, falsifiable claims.

Rules:
1. Each claim must be a single, checkable assertion
2. Split compound sentences into separate claims
3. Ignore opinions unless framed as facts (e.g., "Studies show..." is factual)
4. Preserve the original meaning exactly
5. Include temporal claims (dates, versions, etc.)
6. Mark each claim with your confidence that it's a factual assertion (0.0-1.0)
7. Identify the claim type:
   - DIRECT: Simple, directly verifiable ("X is Y")
   - HEDGED: Contains hedging language ("might", "possibly", "believed to")
   - MULTI_HOP: Requires chaining facts ("A because B and C")
   - TEMPORAL: Time-sensitive ("as of 2023", "recently")
   - COMPARATIVE: Comparison ("faster than", "better than")
   - QUANTITATIVE: Contains numbers/statistics

Text to analyze:
\"\"\"
{text}
\"\"\"

Extract all claims and respond with a JSON object containing a "claims" array.
Each claim object must have:
- "text": the claim text (string)
- "start": character offset where claim starts in original text (integer)
- "end": character offset where claim ends in original text (integer)
- "confidence": your confidence this is a factual claim, not opinion (float 0-1)
- "is_factual": whether this is a factual claim vs opinion (boolean)
- "claim_type": one of DIRECT, HEDGED, MULTI_HOP, TEMPORAL, COMPARATIVE, QUANTITATIVE
- "extraction_confidence": confidence the claim was correctly extracted as atomic (float 0-1)

Example output:
{{
  "claims": [
    {{"text": "Python was created in 1991", "start": 0, "end": 26, "confidence": 0.95, "is_factual": true, "claim_type": "TEMPORAL", "extraction_confidence": 0.98}},
    {{"text": "Python might be the best language", "start": 28, "end": 61, "confidence": 0.3, "is_factual": false, "claim_type": "HEDGED", "extraction_confidence": 0.85}}
  ]
}}"""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "is_factual": {"type": "boolean"},
                    "claim_type": {"type": "string", "enum": ["DIRECT", "HEDGED", "MULTI_HOP", "TEMPORAL", "COMPARATIVE", "QUANTITATIVE"]},
                    "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["text", "start", "end", "confidence", "is_factual"],
            },
        }
    },
    "required": ["claims"],
}


class LLMClaimExtractor(ClaimExtractor):
    """Extracts claims using an LLM with deterministic prompting."""

    def __init__(self, llm_provider: Any, config: ExtractionConfig | None = None):
        self.llm = llm_provider
        self.config = config or ExtractionConfig()
        self._hedging_regex = re.compile("|".join(HEDGING_PATTERNS), re.IGNORECASE)
        self._multi_hop_regex = re.compile("|".join(MULTI_HOP_PATTERNS), re.IGNORECASE)
        self._temporal_regex = re.compile("|".join(TEMPORAL_PATTERNS), re.IGNORECASE)
        self._comparative_regex = re.compile("|".join(COMPARATIVE_PATTERNS), re.IGNORECASE)
        self._quantitative_regex = re.compile("|".join(QUANTITATIVE_PATTERNS), re.IGNORECASE)

    def _generate_claim_id(self, text: str, start: int) -> str:
        """Generate deterministic ID for a claim."""
        content = f"{text}:{start}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _detect_claim_type(self, text: str) -> ClaimType:
        """Detect claim type using rule-based patterns."""
        # Order matters - check more specific patterns first
        if self._hedging_regex.search(text):
            return ClaimType.HEDGED
        if self._multi_hop_regex.search(text):
            return ClaimType.MULTI_HOP
        if self._quantitative_regex.search(text):
            return ClaimType.QUANTITATIVE
        if self._comparative_regex.search(text):
            return ClaimType.COMPARATIVE
        if self._temporal_regex.search(text):
            return ClaimType.TEMPORAL
        return ClaimType.DIRECT

    def _detect_hedging(self, text: str) -> bool:
        """Check if text contains hedging language."""
        return bool(self._hedging_regex.search(text))

    def _generate_claim_id(self, text: str, start: int) -> str:
        """Generate deterministic ID for a claim."""
        content = f"{text}:{start}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _validate_spans(self, claims_data: list[dict], original_text: str) -> list[dict]:
        """Validate and fix claim spans against original text."""
        validated = []
        for claim in claims_data:
            text = claim["text"]
            start = claim.get("start", 0)
            end = claim.get("end", len(text))

            # Try to find the claim text in the original
            found_start = original_text.lower().find(text.lower())
            if found_start >= 0:
                start = found_start
                end = found_start + len(text)
            else:
                # Fuzzy match - find best substring match
                words = text.split()[:5]  # First 5 words
                pattern = r"\b" + r"\s+".join(re.escape(w) for w in words)
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    start = match.start()
                    # Find end of sentence
                    end_match = re.search(r"[.!?]", original_text[start:])
                    end = start + (end_match.end() if end_match else len(text))

            validated.append({**claim, "start": start, "end": min(end, len(original_text))})

        return validated

    def extract(self, text: str) -> list[Claim]:
        """Extract claims from text."""
        claims, _ = self.extract_with_confidence(text)
        return claims

    def extract_with_confidence(self, text: str) -> tuple[list[Claim], dict]:
        """Extract claims with metadata about extraction quality."""
        if not text.strip():
            return [], {"error": "Empty input text"}

        prompt = EXTRACTION_PROMPT.format(text=text)

        # Retry logic for robustness
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                result = self.llm.complete_json(prompt, EXTRACTION_SCHEMA)
                break
            except Exception as e:
                last_error = e
                continue
        else:
            return [], {"error": f"Extraction failed after {self.config.max_retries} attempts: {last_error}"}

        claims_data = result.get("claims", [])

        # Validate spans
        claims_data = self._validate_spans(claims_data, text)

        # Filter and convert to Claim objects
        claims = []
        for i, c in enumerate(claims_data):
            # Skip opinions if configured
            if not self.config.include_opinions and not c.get("is_factual", True):
                continue

            # Skip too short claims
            if len(c["text"]) < self.config.min_claim_length:
                continue

            # Enforce max claims
            if len(claims) >= self.config.max_claims:
                break

            # Detect claim type (use LLM's if provided, else rule-based)
            llm_claim_type = c.get("claim_type", "DIRECT")
            try:
                claim_type = ClaimType(llm_claim_type)
            except ValueError:
                claim_type = self._detect_claim_type(c["text"])

            # Rule-based hedging detection (more reliable than LLM)
            hedging_detected = self._detect_hedging(c["text"])

            # If hedging detected but LLM didn't catch it, override type
            if hedging_detected and claim_type != ClaimType.HEDGED:
                claim_type = ClaimType.HEDGED

            claim = Claim(
                id=self._generate_claim_id(c["text"], c["start"]),
                text=c["text"],
                source_span=(c["start"], c["end"]),
                raw_confidence=c["confidence"],
                is_factual=c.get("is_factual", True),
                claim_type=claim_type,
                extraction_confidence=c.get("extraction_confidence", 0.9),
                hedging_detected=hedging_detected,
            )
            claims.append(claim)

        metadata = {
            "total_extracted": len(claims_data),
            "after_filtering": len(claims),
            "filtered_opinions": len([c for c in claims_data if not c.get("is_factual", True)]),
            "hedged_claims": len([c for c in claims if c.hedging_detected]),
            "claim_types": {ct.value: len([c for c in claims if c.claim_type == ct]) for ct in ClaimType},
        }

        return claims, metadata
