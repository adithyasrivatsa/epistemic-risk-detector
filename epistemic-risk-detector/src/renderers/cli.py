"""CLI renderer with colored output using Rich."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.interfaces import OutputRenderer
from src.core.schemas import AlignmentLabel, AnalysisResult, Verdict, VerdictLabel


class CLIRenderer(OutputRenderer):
    """Renders analysis results for terminal output."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def _verdict_color(self, label: VerdictLabel) -> str:
        """Get color for verdict label."""
        return {
            VerdictLabel.GROUNDED: "green",
            VerdictLabel.WEAK: "yellow",
            VerdictLabel.HALLUCINATED: "red",
        }[label]

    def _alignment_color(self, label: AlignmentLabel) -> str:
        """Get color for alignment label."""
        return {
            AlignmentLabel.SUPPORTS: "green",
            AlignmentLabel.WEAK_SUPPORT: "yellow",
            AlignmentLabel.CONTRADICTS: "red",
            AlignmentLabel.IRRELEVANT: "dim",
        }[label]

    def _risk_bar(self, risk: float, width: int = 20) -> Text:
        """Create a visual risk bar."""
        filled = int(risk * width)
        bar = Text()

        if risk < 0.3:
            color = "green"
        elif risk < 0.7:
            color = "yellow"
        else:
            color = "red"

        bar.append("█" * filled, style=color)
        bar.append("░" * (width - filled), style="dim")
        bar.append(f" {risk:.0%}", style=color)

        return bar

    def render_verdict(self, verdict: Verdict) -> str:
        """Render a single verdict as a panel."""
        color = self._verdict_color(verdict.label)

        # Build content
        content = Text()

        # Verdict and risk
        content.append("Verdict: ", style="bold")
        content.append(f"{verdict.label.value}\n", style=f"bold {color}")

        content.append("Risk Score: ")
        content.append(self._risk_bar(verdict.hallucination_risk))
        content.append("\n\n")

        # Confidence breakdown
        cal = verdict.calibrated_confidence
        content.append("Model Confidence: ", style="bold")
        content.append(f"{cal.raw_confidence:.2f}", style="cyan")
        content.append(" (raw) → ")
        content.append(f"{cal.calibrated_confidence:.2f}", style="cyan")
        content.append(" (calibrated)\n")

        content.append("Evidence Strength: ", style="bold")
        content.append(f"{verdict.evidence_strength:.2f}\n\n", style="cyan")

        # Evidence found
        if verdict.alignments:
            content.append("Evidence Found:\n", style="bold")
            for alignment in verdict.alignments[:3]:  # Show top 3
                label_color = self._alignment_color(alignment.label)
                content.append(f"  [{alignment.label.value}] ", style=label_color)

                # Truncate explanation
                explanation = alignment.explanation[:60]
                if len(alignment.explanation) > 60:
                    explanation += "..."
                content.append(f"{explanation}\n")
            content.append("\n")
        else:
            content.append("Evidence Found: ", style="bold")
            content.append("None\n\n", style="dim")

        # Explanation
        content.append("Why ", style="bold")
        content.append(f"{verdict.label.value}", style=f"bold {color}")
        content.append(":\n", style="bold")
        content.append(f"  {verdict.explanation}")

        # Create panel
        panel = Panel(
            content,
            title=f"[bold]CLAIM: \"{verdict.claim.text[:50]}{'...' if len(verdict.claim.text) > 50 else ''}\"[/bold]",
            border_style=color,
            padding=(1, 2),
        )

        # Render to string
        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    def render(self, result: AnalysisResult) -> str:
        """Render complete analysis result."""
        output_parts = []

        # Header
        header = Text()
        header.append("\n═══ HALLUCINATION ANALYSIS ═══\n\n", style="bold blue")
        header.append(f"Claims analyzed: {len(result.claims)}\n")
        header.append("Overall risk: ")
        header.append(self._risk_bar(result.overall_hallucination_risk))
        header.append("\n")

        with self.console.capture() as capture:
            self.console.print(header)
        output_parts.append(capture.get())

        # Summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Verdict", style="bold")
        table.add_column("Count")
        table.add_column("Percentage")

        verdict_counts = {
            VerdictLabel.GROUNDED: 0,
            VerdictLabel.WEAK: 0,
            VerdictLabel.HALLUCINATED: 0,
        }
        for v in result.verdicts:
            verdict_counts[v.label] += 1

        total = len(result.verdicts) or 1
        for label, count in verdict_counts.items():
            color = self._verdict_color(label)
            table.add_row(
                Text(label.value, style=color),
                str(count),
                f"{count/total:.0%}",
            )

        with self.console.capture() as capture:
            self.console.print(table)
            self.console.print()
        output_parts.append(capture.get())

        # Individual verdicts
        for verdict in result.verdicts:
            output_parts.append(self.render_verdict(verdict))

        # Summary
        with self.console.capture() as capture:
            self.console.print(f"\n[bold]Summary:[/bold] {result.summary}")
        output_parts.append(capture.get())

        return "".join(output_parts)
