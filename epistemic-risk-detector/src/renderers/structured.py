"""Structured output renderer for JSON and web-ready formats."""

import json
from typing import Any

from src.core.interfaces import OutputRenderer
from src.core.schemas import AnalysisResult, Verdict


class StructuredRenderer(OutputRenderer):
    """Renders analysis results as structured JSON for web/API consumption."""

    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def _verdict_to_dict(self, verdict: Verdict) -> dict[str, Any]:
        """Convert verdict to web-ready dictionary."""
        return {
            "claim": {
                "id": verdict.claim.id,
                "text": verdict.claim.text,
                "span": list(verdict.claim.source_span),
                "is_factual": verdict.claim.is_factual,
            },
            "verdict": verdict.label.value,
            "hallucination_risk": round(verdict.hallucination_risk, 3),
            "evidence_strength": round(verdict.evidence_strength, 3),
            "confidence": {
                "raw": round(verdict.calibrated_confidence.raw_confidence, 3),
                "calibrated": round(verdict.calibrated_confidence.calibrated_confidence, 3),
                "penalties": verdict.calibrated_confidence.penalties_applied,
            },
            "contradiction_detected": verdict.contradiction_detected,
            "explanation": verdict.explanation,
            "alignments": [
                {
                    "evidence_id": a.evidence_id,
                    "label": a.label.value,
                    "confidence": round(a.confidence, 3),
                    "explanation": a.explanation,
                    "temporal_match": a.temporal_match,
                    "scores": {
                        "semantic": round(a.semantic_score, 3),
                        "logical": round(a.logical_score, 3),
                    },
                }
                for a in verdict.alignments
            ],
            "best_evidence": (
                {
                    "id": verdict.best_evidence.id,
                    "text": verdict.best_evidence.text,
                    "source": verdict.best_evidence.source,
                    "similarity": round(verdict.best_evidence.similarity_score, 3),
                }
                if verdict.best_evidence
                else None
            ),
        }

    def render_verdict(self, verdict: Verdict) -> str:
        """Render a single verdict as JSON."""
        data = self._verdict_to_dict(verdict)
        if self.pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)

    def render(self, result: AnalysisResult) -> str:
        """Render complete analysis result as JSON."""
        data = {
            "original_text": result.original_text,
            "claims_count": len(result.claims),
            "overall_hallucination_risk": round(result.overall_hallucination_risk, 3),
            "summary": result.summary,
            "verdicts": [self._verdict_to_dict(v) for v in result.verdicts],
            "statistics": {
                "grounded": sum(1 for v in result.verdicts if v.label.value == "GROUNDED"),
                "weak": sum(1 for v in result.verdicts if v.label.value == "WEAK"),
                "hallucinated": sum(1 for v in result.verdicts if v.label.value == "HALLUCINATED"),
            },
            "metadata": result.metadata,
        }

        if self.pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)

    def render_for_web(self, result: AnalysisResult) -> dict[str, Any]:
        """Return structured data directly (for web frameworks)."""
        return {
            "original_text": result.original_text,
            "claims_count": len(result.claims),
            "overall_hallucination_risk": round(result.overall_hallucination_risk, 3),
            "summary": result.summary,
            "verdicts": [self._verdict_to_dict(v) for v in result.verdicts],
            "statistics": {
                "grounded": sum(1 for v in result.verdicts if v.label.value == "GROUNDED"),
                "weak": sum(1 for v in result.verdicts if v.label.value == "WEAK"),
                "hallucinated": sum(1 for v in result.verdicts if v.label.value == "HALLUCINATED"),
            },
            # Include highlighted text with claim spans for UI
            "highlighted_claims": [
                {
                    "start": v.claim.source_span[0],
                    "end": v.claim.source_span[1],
                    "verdict": v.label.value,
                    "claim_id": v.claim.id,
                }
                for v in result.verdicts
            ],
        }
