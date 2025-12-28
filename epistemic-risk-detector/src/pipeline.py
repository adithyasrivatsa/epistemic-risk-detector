"""Main analysis pipeline - orchestrates all modules."""

from src.calibrators.confidence import PenaltyBasedCalibrator
from src.core.config import Config, load_config
from src.core.schemas import AnalysisResult, VerdictLabel
from src.evaluators.alignment import LLMAlignmentEvaluator
from src.extractors.claim_extractor import LLMClaimExtractor
from src.providers.llm import LLMProviderFactory
from src.renderers.cli import CLIRenderer
from src.renderers.structured import StructuredRenderer
from src.retrievers.local_vector import LocalVectorStore
from src.verdict.engine import DefaultVerdictEngine


class EpistemicRiskDetector:
    """Main pipeline for epistemic risk detection."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()

        # Initialize components
        self.llm = LLMProviderFactory.create(self.config.llm)
        self.extractor = LLMClaimExtractor(self.llm, self.config.extraction)
        self.retriever = LocalVectorStore(self.config.retrieval)
        self.evaluator = LLMAlignmentEvaluator(self.llm)
        self.calibrator = PenaltyBasedCalibrator(self.config.calibration)
        self.verdict_engine = DefaultVerdictEngine(self.config.verdict)

        # Renderers
        self.cli_renderer = CLIRenderer()
        self.json_renderer = StructuredRenderer()

    def index_corpus(self, path: str, extensions: list[str] | None = None) -> int:
        """Index documents for evidence retrieval."""
        from pathlib import Path

        p = Path(path)
        if p.is_file():
            return self.retriever.index_document(str(p))
        return self.retriever.index_directory(str(p), extensions)

    def analyze(self, text: str) -> AnalysisResult:
        """
        Analyze an LLM response for hallucinations.

        Args:
            text: The LLM response to analyze

        Returns:
            Complete analysis result with verdicts for each claim
        """
        # Step 1: Extract claims
        claims, extraction_meta = self.extractor.extract_with_confidence(text)

        if not claims:
            return AnalysisResult(
                original_text=text,
                claims=[],
                verdicts=[],
                overall_hallucination_risk=0.0,
                summary="No factual claims found in the text.",
                metadata={"extraction": extraction_meta},
            )

        verdicts = []

        # Step 2-5: Process each claim
        for claim in claims:
            # Retrieve evidence
            evidence = self.retriever.retrieve(claim.text)

            # Evaluate alignment
            alignments = self.evaluator.evaluate(claim, evidence)

            # Calibrate confidence
            calibrated = self.calibrator.calibrate(claim, alignments, evidence)

            # Compute verdict
            verdict = self.verdict_engine.compute(claim, evidence, alignments, calibrated)
            verdicts.append(verdict)

        # Compute overall risk
        if verdicts:
            overall_risk = sum(v.hallucination_risk for v in verdicts) / len(verdicts)
        else:
            overall_risk = 0.0

        # Generate summary
        hallucinated_count = sum(1 for v in verdicts if v.label == VerdictLabel.HALLUCINATED)
        grounded_count = sum(1 for v in verdicts if v.label == VerdictLabel.GROUNDED)

        if hallucinated_count == 0:
            summary = f"All {len(verdicts)} claims appear grounded or weakly supported."
        elif hallucinated_count == len(verdicts):
            summary = f"All {len(verdicts)} claims appear to be hallucinations."
        else:
            summary = (
                f"{hallucinated_count}/{len(verdicts)} claims flagged as potential hallucinations. "
                f"{grounded_count} claims are well-grounded."
            )

        return AnalysisResult(
            original_text=text,
            claims=claims,
            verdicts=verdicts,
            overall_hallucination_risk=overall_risk,
            summary=summary,
            metadata={
                "extraction": extraction_meta,
                "corpus_stats": self.retriever.stats(),
            },
        )

    def render_cli(self, result: AnalysisResult) -> str:
        """Render result for CLI output."""
        return self.cli_renderer.render(result)

    def render_json(self, result: AnalysisResult) -> str:
        """Render result as JSON."""
        return self.json_renderer.render(result)

    def render_web(self, result: AnalysisResult) -> dict:
        """Render result for web consumption."""
        return self.json_renderer.render_for_web(result)
