#!/usr/bin/env python3
"""
THE KILLER DEMO: "Python 3.12 Removed the GIL"

This demonstrates the epistemic risk detector on a perfect edge case:
- Partially true: PEP 703 was accepted, per-interpreter GIL exists
- Technically false: GIL was NOT removed in 3.12
- Commonly hallucinated: LLMs confidently state this incorrectly
- Nuanced: Requires understanding the difference between "optional" and "removed"

Run this to see exactly how the system handles hallucination-adjacent claims.

NOTE: This is a demo showing the output format. In production, claims are 
extracted and evaluated using your LLM against your corpus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibrators.confidence import PenaltyBasedCalibrator
from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    AnalysisResult,
    CalibratedConfidence,
    Claim,
    ClaimType,
    ContradictionType,
    EvidenceChunk,
    Verdict,
    VerdictLabel,
)
from src.renderers.cli import CLIRenderer
from src.renderers.structured import StructuredRenderer


def create_gil_demo() -> AnalysisResult:
    """
    Create the GIL removal hallucination demo.
    
    This is the canonical example showing:
    1. High semantic similarity (same topic)
    2. Direct negation contradiction
    3. Nuanced partial truth
    4. Proper penalty application
    """
    
    original_text = (
        "Python 3.12 completely removed the Global Interpreter Lock, "
        "enabling true multi-threaded execution. This was a major milestone "
        "that the Python community had been waiting for since the language's creation."
    )
    
    # Claim 1: The main hallucination
    claim1 = Claim(
        id="gil_claim_001",
        text="Python 3.12 completely removed the Global Interpreter Lock",
        source_span=(0, 59),
        raw_confidence=0.92,  # LLMs are very confident about this
        is_factual=True,
        claim_type=ClaimType.TEMPORAL,
        extraction_confidence=0.95,
        hedging_detected=False,
    )
    
    # Evidence that CONTRADICTS - note the explicit "did NOT"
    evidence1 = EvidenceChunk(
        id="ev_gil_001",
        text=(
            "Python 3.12 did NOT remove the GIL - it introduced per-interpreter GIL "
            "as an experimental feature. The GIL still exists and is enabled by default."
        ),
        source="python_facts.txt",
        similarity_score=0.88,  # High similarity - same topic!
        chunk_index=5,
        metadata={"date": "2023-10"},
    )
    
    # Evidence that provides context (weak support for the general direction)
    evidence2 = EvidenceChunk(
        id="ev_gil_002",
        text=(
            "PEP 703, titled 'Making the Global Interpreter Lock Optional in CPython', "
            "was accepted in July 2023. This PEP proposes adding a build configuration "
            "option to disable the GIL, making it optional rather than removing it entirely."
        ),
        source="python_facts.txt",
        similarity_score=0.75,
        chunk_index=4,
        metadata={"date": "2023-07"},
    )
    
    # Alignment 1: CONTRADICTS with DIRECT_NEGATION
    alignment1 = AlignmentResult(
        claim_id=claim1.id,
        evidence_id=evidence1.id,
        label=AlignmentLabel.CONTRADICTS,
        confidence=0.94,
        explanation=(
            "Evidence explicitly states 'did NOT remove the GIL'. "
            "The claim asserts complete removal, evidence confirms GIL still exists."
        ),
        temporal_match=True,  # Both reference 3.12
        semantic_score=0.88,  # High - same topic
        logical_score=0.08,   # Low - logical contradiction
        contradiction_type=ContradictionType.DIRECT_NEGATION,
        negation_detected=True,
        claim_date="3.12",
        evidence_date="3.12",
    )
    
    # Alignment 2: WEAK_SUPPORT - shows the nuance
    alignment2 = AlignmentResult(
        claim_id=claim1.id,
        evidence_id=evidence2.id,
        label=AlignmentLabel.WEAK_SUPPORT,
        confidence=0.65,
        explanation=(
            "PEP 703 shows movement toward optional GIL, but 'optional' != 'removed'. "
            "The claim overstates the actual change."
        ),
        temporal_match=False,  # PEP is 2023, claim is about 3.12 specifically
        semantic_score=0.75,
        logical_score=0.35,
        contradiction_type=ContradictionType.NONE,
        negation_detected=False,
        claim_date="3.12",
        evidence_date="July 2023",
    )
    
    # Calibrated confidence - watch the penalty stack
    calibrated1 = CalibratedConfidence(
        claim_id=claim1.id,
        raw_confidence=0.92,
        calibrated_confidence=0.32,  # Massive drop due to contradiction
        penalties_applied=["contradiction_detected"],
        penalty_breakdown={"contradiction_detected": 0.60},
    )
    
    # Final verdict
    verdict1 = Verdict(
        claim=claim1,
        label=VerdictLabel.HALLUCINATED,
        hallucination_risk=0.89,
        evidence_strength=0.12,  # Low despite high similarity - contradiction tanks it
        calibrated_confidence=calibrated1,
        alignments=[alignment1, alignment2],
        best_evidence=evidence1,
        contradiction_detected=True,
        explanation=(
            "High confidence (0.92) with contradicting evidence. "
            "Evidence explicitly states 'did NOT remove' - direct negation detected. "
            "While PEP 703 shows progress toward optional GIL, the claim that 3.12 "
            "'completely removed' it is factually incorrect."
        ),
    )
    
    # Claim 2: The follow-on claim (also problematic)
    claim2 = Claim(
        id="gil_claim_002",
        text="This enabled true multi-threaded execution",
        source_span=(61, 103),
        raw_confidence=0.88,
        is_factual=True,
        claim_type=ClaimType.DIRECT,
        extraction_confidence=0.90,
        hedging_detected=False,
    )
    
    # No direct evidence for this claim
    calibrated2 = CalibratedConfidence(
        claim_id=claim2.id,
        raw_confidence=0.88,
        calibrated_confidence=0.48,
        penalties_applied=["no_evidence", "depends_on_false_premise"],
        penalty_breakdown={"no_evidence": 0.40},
    )
    
    verdict2 = Verdict(
        claim=claim2,
        label=VerdictLabel.HALLUCINATED,
        hallucination_risk=0.82,
        evidence_strength=0.0,
        calibrated_confidence=calibrated2,
        alignments=[],
        best_evidence=None,
        contradiction_detected=False,
        explanation=(
            "No evidence found for this claim. Additionally, this claim depends on "
            "the previous false premise that the GIL was removed. Since the GIL still "
            "exists, 'true multi-threaded execution' is not enabled."
        ),
    )
    
    return AnalysisResult(
        original_text=original_text,
        claims=[claim1, claim2],
        verdicts=[verdict1, verdict2],
        overall_hallucination_risk=0.855,
        summary=(
            "2/2 claims flagged as hallucinations. The core claim about GIL removal "
            "is directly contradicted by evidence. This is a common LLM hallucination - "
            "confusing 'PEP accepted' with 'feature shipped' and 'optional' with 'removed'."
        ),
        metadata={
            "demo": "gil_removal",
            "why_this_matters": (
                "This case shows the system correctly handling nuanced, "
                "partially-true claims that LLMs commonly get wrong."
            ),
        },
    )


def main():
    print("=" * 70)
    print("THE KILLER DEMO: Python 3.12 GIL Removal Hallucination")
    print("=" * 70)
    print()
    print("This is the perfect test case because:")
    print("  • LLMs confidently state this incorrectly")
    print("  • It's partially true (PEP 703 exists)")
    print("  • It's technically false (GIL was NOT removed)")
    print("  • High semantic similarity makes naive detection fail")
    print()
    print("Input text:")
    print("-" * 70)
    
    result = create_gil_demo()
    print(f'"{result.original_text}"')
    print("-" * 70)
    print()
    
    # Render CLI output
    cli_renderer = CLIRenderer()
    print(cli_renderer.render(result))
    
    print()
    print("=" * 70)
    print("WHY THIS WORKS")
    print("=" * 70)
    print("""
1. HIGH SIMILARITY, LOW LOGIC SCORE
   - Semantic similarity: 0.88 (same topic - GIL, Python 3.12)
   - Logical score: 0.08 (direct contradiction)
   - Naive embedding-only systems would miss this

2. EXPLICIT NEGATION DETECTION
   - Evidence contains "did NOT remove"
   - System detects negation mismatch
   - Contradiction type: DIRECT_NEGATION

3. NUANCED PARTIAL TRUTH
   - PEP 703 IS real and WAS accepted
   - But "accepted" ≠ "shipped"
   - And "optional" ≠ "removed"
   - System shows WEAK_SUPPORT for context

4. TRANSPARENT PENALTY APPLICATION
   - Raw confidence: 0.92 (LLM is very sure)
   - Contradiction penalty: -0.60
   - Calibrated: 0.32 (now reflects reality)

5. CLEAR EXPLANATION
   - Shows exactly WHY it's hallucinated
   - Cites the contradicting evidence
   - Explains the nuance
""")
    
    print("=" * 70)
    print("JSON OUTPUT (for integration)")
    print("=" * 70)
    json_renderer = StructuredRenderer(pretty=True)
    print(json_renderer.render(result))


if __name__ == "__main__":
    main()
