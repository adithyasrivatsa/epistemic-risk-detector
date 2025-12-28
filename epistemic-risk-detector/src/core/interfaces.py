"""Abstract interfaces for all modules. Each is independently testable and swappable."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from src.core.schemas import (
    AlignmentResult,
    AnalysisResult,
    CalibratedConfidence,
    Claim,
    EvidenceChunk,
    Verdict,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM backends. Implement this to add new model providers."""

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a completion for the given prompt."""
        ...

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Generate a JSON completion conforming to the schema."""
        ...


class ClaimExtractor(ABC):
    """Extracts atomic, falsifiable claims from LLM responses."""

    @abstractmethod
    def extract(self, text: str) -> list[Claim]:
        """
        Extract claims from text.

        Args:
            text: Raw LLM response

        Returns:
            List of atomic claims with confidence scores
        """
        pass

    @abstractmethod
    def extract_with_confidence(self, text: str) -> tuple[list[Claim], dict]:
        """
        Extract claims and return extraction metadata.

        Returns:
            Tuple of (claims, metadata dict with extraction stats)
        """
        pass


class EvidenceProvider(ABC):
    """Interface for evidence retrieval. Implement for different sources."""

    @abstractmethod
    def retrieve(self, claim: str, top_k: int = 5) -> list[EvidenceChunk]:
        """
        Retrieve evidence chunks relevant to a claim.

        Args:
            claim: The claim text to find evidence for
            top_k: Maximum number of chunks to return

        Returns:
            List of evidence chunks, may be empty (valid signal)
        """
        pass

    @abstractmethod
    def index_document(self, path: str) -> int:
        """
        Index a document into the evidence store.

        Returns:
            Number of chunks indexed
        """
        pass

    @abstractmethod
    def index_directory(self, path: str, extensions: list[str] | None = None) -> int:
        """
        Index all documents in a directory.

        Returns:
            Total number of chunks indexed
        """
        pass


class AlignmentEvaluator(ABC):
    """Evaluates semantic and logical alignment between claims and evidence."""

    @abstractmethod
    def evaluate(self, claim: Claim, evidence: list[EvidenceChunk]) -> list[AlignmentResult]:
        """
        Evaluate alignment between a claim and evidence chunks.

        Args:
            claim: The claim to evaluate
            evidence: Retrieved evidence chunks

        Returns:
            Alignment results for each (claim, evidence) pair
        """
        pass

    @abstractmethod
    def evaluate_single(self, claim: Claim, evidence: EvidenceChunk) -> AlignmentResult:
        """Evaluate alignment for a single claim-evidence pair."""
        pass


class ConfidenceCalibrator(ABC):
    """Calibrates model confidence based on evidence and language patterns."""

    @abstractmethod
    def calibrate(
        self,
        claim: Claim,
        alignments: list[AlignmentResult],
        evidence: list[EvidenceChunk],
    ) -> CalibratedConfidence:
        """
        Calibrate confidence for a claim.

        Applies penalties for:
        - No evidence found
        - Contradictions detected
        - Vague language in claim

        Returns:
            Calibrated confidence with penalty breakdown
        """
        pass


class VerdictEngine(ABC):
    """Produces final verdicts by combining evidence strength and calibrated confidence."""

    @abstractmethod
    def compute(
        self,
        claim: Claim,
        evidence: list[EvidenceChunk],
        alignments: list[AlignmentResult],
        calibrated: CalibratedConfidence,
    ) -> Verdict:
        """
        Compute final verdict for a claim.

        Returns:
            Verdict with label, risk score, and explanation
        """
        pass


class OutputRenderer(ABC):
    """Renders analysis results for human consumption."""

    @abstractmethod
    def render(self, result: AnalysisResult) -> str:
        """
        Render analysis result as string.

        Returns:
            Formatted output (CLI text, JSON, etc.)
        """
        pass

    @abstractmethod
    def render_verdict(self, verdict: Verdict) -> str:
        """Render a single verdict."""
        pass
