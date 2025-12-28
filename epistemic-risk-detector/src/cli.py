"""Command-line interface for the epistemic risk detector."""

from pathlib import Path

import click
from rich.console import Console

from src.core.config import Config, load_config
from src.pipeline import EpistemicRiskDetector

console = Console()


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.pass_context
def main(ctx: click.Context, config: str | None) -> None:
    """Epistemic Risk Detector - Inspect epistemic risk in LLM outputs."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config) if config else load_config()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--extensions", "-e", multiple=True, help="File extensions to index")
@click.pass_context
def index(ctx: click.Context, path: str, extensions: tuple[str, ...]) -> None:
    """Index documents for evidence retrieval."""
    config: Config = ctx.obj["config"]
    detector = EpistemicRiskDetector(config)

    ext_list = list(extensions) if extensions else None

    with console.status(f"Indexing {path}..."):
        count = detector.index_corpus(path, ext_list)

    console.print(f"[green]✓[/green] Indexed {count} chunks from {path}")
    stats = detector.retriever.stats()
    console.print(f"  Total: {stats['total_chunks']} chunks from {stats['total_documents']} documents")


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def analyze(ctx: click.Context, text: str | None, file: str | None, output_json: bool) -> None:
    """Analyze an LLM response for epistemic risk."""
    config: Config = ctx.obj["config"]

    if file:
        text = Path(file).read_text()
    elif not text:
        console.print("[red]Error:[/red] Provide text as argument or use --file")
        raise SystemExit(1)

    detector = EpistemicRiskDetector(config)

    # Check if corpus is indexed
    stats = detector.retriever.stats()
    if stats["total_chunks"] == 0:
        console.print("[yellow]Warning:[/yellow] No documents indexed. Run 'index' first for evidence retrieval.")
        console.print()

    with console.status("Analyzing..."):
        result = detector.analyze(text)

    if output_json:
        console.print(detector.render_json(result))
    else:
        console.print(detector.render_cli(result))


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show corpus statistics."""
    config: Config = ctx.obj["config"]
    detector = EpistemicRiskDetector(config)

    stats = detector.retriever.stats()
    console.print(f"Indexed chunks: {stats['total_chunks']}")
    console.print(f"Documents: {stats['total_documents']}")
    console.print(f"Database: {config.retrieval.db_path}")


@main.command()
@click.option("--output", "-o", type=click.Path(), default="config.yaml", help="Output path")
@click.pass_context
def init(ctx: click.Context, output: str) -> None:
    """Generate a default configuration file."""
    config = Config()
    config.to_yaml(output)
    console.print(f"[green]✓[/green] Created {output}")


@main.command()
@click.pass_context
def clear(ctx: click.Context) -> None:
    """Clear the indexed corpus."""
    config: Config = ctx.obj["config"]
    detector = EpistemicRiskDetector(config)

    detector.retriever.clear()
    console.print("[green]✓[/green] Corpus cleared")


if __name__ == "__main__":
    main()
