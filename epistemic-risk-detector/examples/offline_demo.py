#!/usr/bin/env python3
"""
Offline demo that works without LLM API access.

Uses mock components to demonstrate the pipeline flow and output format.

NOTE: This shows the output format only. Actual detection quality depends on
your LLM, corpus, and domain-specific calibration.
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
    EvidenceChunk,
    Verdict,
    VerdictLabel,
)
from src.renderers.cli import CLIRenderer
from src.renderers.structured import StructuredRenderer
from src.verdict.engine import DefaultVerdictEngine


def create_demo_analysis() -> AnalysisResult:
    """Create a demo analysis result showing different verdict types."""
    
    # Claim 1: Grounded (correct fact with evidence)
    claim1 = Claim(
        id="claim_001",
        text="Python was created by Guido van Rossum in 1991",
        source_span=(0, 46),
        raw_confidence=0.95,
        is_factual=True,
    )
    
    evidence1 = EvidenceChunk(
        id="ev_001",
        text="Python was created by Guido van Rossum and first released in 1991.",
        source="python_facts.txt",
        similarity_score=0.94,
        chunk_index=0,
    )
    
    alignment1 = AlignmentResult(
        claim_id=claim1.id,
        evidence_id=evidence1.id,
        label=AlignmentLabel.SUPPORTS,
        confidence=0.92,
        explanation="Evidence directly confirms Python's creation date and creator",
        temporal_match=True,
        semantic_score=0.94,
        logical_score=0.90,
    )
    
    calibrated1 = CalibratedConfidence(
        claim_id=claim1.id,
        raw_confidence=0.95,
        calibrated_confidence=0.93,
        penalties_applied=[],
        penalty_breakdown={},
    )
    
    verdict1 = Verdict(
        claim=claim1,
        label=VerdictLabel.GROUNDED,
        hallucination_risk=0.12,
        evidence_strength=0.85,
        calibrated_confidence=calibrated1,
        alignments=[alignment1],
        best_evidence=evidence1,
        contradiction_detected=False,
        explanation="Strong evidence supports this claim (strength: 0.85). 1 evidence chunk(s) directly support.",
    )
    
    # Claim 2: Hallucinated (incorrect fact with contradicting evidence)
    claim2 = Claim(
        id="claim_002",
        text="Python 3.12 removed the GIL completely",
        source_span=(48, 86),
        raw_confidence=0.92,
        is_factual=True,
    )
    
    evidence2 = EvidenceChunk(
        id="ev_002",
        text="Python 3.12 did NOT remove the GIL - it introduced per-interpreter GIL as an experimental feature.",
        source="python_facts.txt",
        similarity_score=0.88,
        chunk_index=5,
    )
    
    alignment2 = AlignmentResult(
        claim_id=claim2.id,
        evidence_id=evidence2.id,
        label=AlignmentLabel.CONTRADICTS,
        confidence=0.91,
        explanation="Evidence explicitly states GIL was NOT removed in Python 3.12",
        temporal_match=True,
        semantic_score=0.88,
        logical_score=0.15,
    )
    
    calibrated2 = CalibratedConfidence(
        claim_id=claim2.id,
        raw_confidence=0.92,
        calibrated_confidence=0.32,
        penalties_applied=["contradiction_detected"],
        penalty_breakdown={"contradiction_detected": 0.6},
    )
    
    verdict2 = Verdict(
        claim=claim2,
        label=VerdictLabel.HALLUCINATED,
        hallucination_risk=0.89,
        evidence_strength=0.12,
        calibrated_confidence=calibrated2,
        alignments=[alignment2],
        best_evidence=evidence2,
        contradiction_detected=True,
        explanation="High confidence (0.92) with contradicting evidence. Best evidence contradicts the claim.",
    )
    
    # Claim 3: Weak (partial support)
    claim3 = Claim(
        id="claim_003",
        text="GPT-4 was released in early 2023",
        source_span=(88, 120),
        raw_confidence=0.85,
        is_factual=True,
    )
    
    evidence3 = EvidenceChunk(
        id="ev_003",
        text="GPT-4, released in March 2023, is a multimodal model.",
        source="ml_facts.txt",
        similarity_score=0.72,
        chunk_index=4,
    )
    
    alignment3 = AlignmentResult(
        claim_id=claim3.id,
        evidence_id=evidence3.id,
        label=AlignmentLabel.WEAK_SUPPORT,
        confidence=0.65,
        explanation="Evidence confirms March 2023 release, which is 'early 2023' but not exact",
        temporal_match=True,
        semantic_score=0.72,
        logical_score=0.60,
    )
    
    calibrated3 = CalibratedConfidence(
        claim_id=claim3.id,
        raw_confidence=0.85,
        calibrated_confidence=0.70,
        penalties_applied=["weak_evidence_only"],
        penalty_breakdown={"weak_evidence_only": 0.15},
    )
    
    verdict3 = Verdict(
        claim=claim3,
        label=VerdictLabel.WEAK,
        hallucination_risk=0.45,
        evidence_strength=0.48,
        calibrated_confidence=calibrated3,
        alignments=[alignment3],
        best_evidence=evidence3,
        contradiction_detected=False,
        explanation="Partial support found (evidence strength: 0.48). Confidence reduced from 0.85 to 0.70.",
    )
    
    return AnalysisResult(
        original_text="Python was created by Guido van Rossum in 1991. Python 3.12 removed the GIL completely. GPT-4 was released in early 2023.",
        claims=[claim1, claim2, claim3],
        verdicts=[verdict1, verdict2, verdict3],
        overall_hallucination_risk=0.49,
        summary="1/3 claims flagged as potential hallucinations. 1 claims are well-grounded.",
        metadata={"demo": True},
    )


def main():
    print("=" * 70)
    print("LLM EPISTEMIC RISK DETECTOR - OFFLINE DEMO")
    print("=" * 70)
    print()
    print("This demo shows the output format without requiring LLM API access.")
    print("In production, claims are extracted and evaluated using your LLM.")
    print("Verdicts are signals, not judgments.")
    print()
    
    # Create demo result
    result = create_demo_analysis()
    
    # Render CLI output
    cli_renderer = CLIRenderer()
    print(cli_renderer.render(result))
    
    print("\n" + "=" * 70)
    print("JSON OUTPUT (for web/API integration)")
    print("=" * 70 + "\n")
    
    # Render JSON output
    json_renderer = StructuredRenderer(pretty=True)
    print(json_renderer.render(result))


if __name__ == "__main__":
    main()
