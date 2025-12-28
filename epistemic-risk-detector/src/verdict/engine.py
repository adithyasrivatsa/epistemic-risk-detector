"""Verdict engine - combines evidence strength and calibrated confidence."""

from src.core.config import VerdictConfig
from src.core.interfaces import VerdictEngine
from src.core.schemas import (
    AlignmentLabel,
    AlignmentResult,
    CalibratedConfidence,
    Claim,
    EvidenceChunk,
    Verdict,
    VerdictLabel,
)


class DefaultVerdictEngine(VerdictEngine):
    """Produces verdicts by combining evidence strength and calibrated confidence."""

    def __init__(self, config: VerdictConfig | None = None):
        self.config = config or VerdictConfig()

    def _compute_evidence_strength(
        self, alignments: list[AlignmentResult], evidence: list[EvidenceChunk]
    ) -> float:
        """Compute normalized evidence strength score."""
        if not alignments:
            return 0.0

        # Score based on best alignment
        label_scores = {
            AlignmentLabel.SUPPORTS: 1.0,
            AlignmentLabel.WEAK_SUPPORT: 0.5,
            AlignmentLabel.CONTRADICTS: 0.1,  # Some evidence exists, but contradicts
            AlignmentLabel.IRRELEVANT: 0.0,
        }

        scores = []
        for alignment in alignments:
            base_score = label_scores[alignment.label]

            # Weight by alignment confidence and semantic/logical scores
            weighted = (
                base_score
                * alignment.confidence
                * (alignment.semantic_score + alignment.logical_score) / 2
            )

            # Temporal mismatch penalty
            if not alignment.temporal_match:
                weighted *= 0.7

            scores.append(weighted)

        # Return best score (most supportive evidence)
        return max(scores) if scores else 0.0

    def _find_best_evidence(
        self, alignments: list[AlignmentResult], evidence: list[EvidenceChunk]
    ) -> EvidenceChunk | None:
        """Find the most relevant evidence chunk."""
        if not alignments or not evidence:
            return None

        # Prefer supporting evidence, then weak support
        for label in [AlignmentLabel.SUPPORTS, AlignmentLabel.WEAK_SUPPORT]:
            for alignment in alignments:
                if alignment.label == label:
                    for e in evidence:
                        if e.id == alignment.evidence_id:
                            return e

        # Fall back to highest similarity
        return max(evidence, key=lambda e: e.similarity_score)

    def _generate_explanation(
        self,
        claim: Claim,
        label: VerdictLabel,
        evidence_strength: float,
        calibrated: CalibratedConfidence,
        alignments: list[AlignmentResult],
        contradiction_detected: bool,
    ) -> str:
        """Generate human-readable explanation for the verdict."""
        parts = []

        if label == VerdictLabel.HALLUCINATED:
            parts.append(f"High confidence ({calibrated.raw_confidence:.2f}) with ")
            if not alignments:
                parts.append("no supporting evidence found.")
            elif contradiction_detected:
                parts.append("contradicting evidence.")
            else:
                parts.append(f"weak evidence (strength: {evidence_strength:.2f}).")

            if calibrated.penalties_applied:
                parts.append(f" Penalties: {', '.join(calibrated.penalties_applied)}.")

        elif label == VerdictLabel.WEAK:
            parts.append(
                f"Partial support found (evidence strength: {evidence_strength:.2f}). "
            )
            if calibrated.calibrated_confidence < calibrated.raw_confidence:
                parts.append(
                    f"Confidence reduced from {calibrated.raw_confidence:.2f} to "
                    f"{calibrated.calibrated_confidence:.2f}."
                )

        else:  # GROUNDED
            parts.append(
                f"Strong evidence supports this claim (strength: {evidence_strength:.2f}). "
            )
            if alignments:
                supporting = [a for a in alignments if a.label == AlignmentLabel.SUPPORTS]
                if supporting:
                    parts.append(f"{len(supporting)} evidence chunk(s) directly support.")

        return "".join(parts)

    def compute(
        self,
        claim: Claim,
        evidence: list[EvidenceChunk],
        alignments: list[AlignmentResult],
        calibrated: CalibratedConfidence,
    ) -> Verdict:
        """Compute final verdict for a claim."""
        evidence_strength = self._compute_evidence_strength(alignments, evidence)
        contradiction_detected = any(
            a.label == AlignmentLabel.CONTRADICTS for a in alignments
        )

        # Compute hallucination risk
        # High confidence + low evidence = high risk
        confidence_factor = calibrated.raw_confidence  # Use raw, not calibrated
        evidence_factor = 1.0 - evidence_strength

        # Weight evidence more heavily
        hallucination_risk = (
            self.config.confidence_weight * confidence_factor
            + self.config.evidence_weight * evidence_factor
        )

        # Boost risk if contradiction detected
        if contradiction_detected:
            hallucination_risk = min(1.0, hallucination_risk + 0.2)

        # Determine verdict label
        if evidence_strength >= self.config.grounded_threshold and not contradiction_detected:
            label = VerdictLabel.GROUNDED
        elif evidence_strength <= self.config.hallucination_threshold or contradiction_detected:
            label = VerdictLabel.HALLUCINATED
        else:
            label = VerdictLabel.WEAK

        explanation = self._generate_explanation(
            claim, label, evidence_strength, calibrated, alignments, contradiction_detected
        )

        return Verdict(
            claim=claim,
            label=label,
            hallucination_risk=hallucination_risk,
            evidence_strength=evidence_strength,
            calibrated_confidence=calibrated,
            alignments=alignments,
            best_evidence=self._find_best_evidence(alignments, evidence),
            contradiction_detected=contradiction_detected,
            explanation=explanation,
        )
