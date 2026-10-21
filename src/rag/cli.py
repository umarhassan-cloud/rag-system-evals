from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag.config import get_config, get_settings
from rag.eval import render_comparison_table, run_eval
from rag.llm import build_llm_endpoint
from rag.pipeline import RAGPipeline
from rag.retrieval import STRATEGIES

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Document Q&A RAG with retrieval evals")


def _console() -> Console:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    return Console(force_terminal=True)


console = _console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="File or directory to ingest"),
    reset: bool = typer.Option(False, "--reset", help="Clear existing index before ingesting"),
) -> None:
    """Ingest documents into the vector store."""
    pipeline = RAGPipeline()
    stats = pipeline.ingest([path], reset=reset)
    console.print(
        Panel.fit(
            f"Ingested [bold]{stats['chunks']}[/bold] chunks\n"
            f"Collection size: [bold]{stats['collection_size']}[/bold]",
            title="Ingest complete",
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    strategy: str = typer.Option("hybrid", "--strategy", "-s", help=f"One of: {', '.join(STRATEGIES)}"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k"),
) -> None:
    """Ask a question and print an answer with citations."""
    if strategy not in STRATEGIES:
        raise typer.BadParameter(f"strategy must be one of {STRATEGIES}")

    settings = get_settings()
    endpoint = build_llm_endpoint(settings)
    if endpoint.provider == "mock":
        console.print(
            "[yellow]No free LLM key found — using grounded mock answers.\n"
            "Set GROQ_API_KEY (recommended) or HF_TOKEN in .env. "
            "See `rag providers`.[/yellow]"
        )

    result = RAGPipeline().ask(question, strategy=strategy, top_k=top_k)  # type: ignore[arg-type]
    console.print(Panel(result.answer, title=f"Answer ({result.strategy} / {result.model})"))

    table = Table(title="Citations", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Source")
    table.add_column("Score", justify="right")
    table.add_column("Excerpt")
    for c in result.citations:
        table.add_row(str(c.index), c.source, f"{c.score:.3f}", c.excerpt)
    console.print(table)


@app.command(name="eval")
def eval_cmd(
    strategies: str = typer.Option(
        ",".join(STRATEGIES),
        "--strategies",
        help="Comma-separated strategies to compare",
    ),
    dataset: Optional[Path] = typer.Option(None, "--dataset", exists=True),
    no_answers: bool = typer.Option(False, "--no-answers", help="Score retrieval only"),
) -> None:
    """Run evals and print a retrieval-strategy comparison table."""
    chosen = [s.strip() for s in strategies.split(",") if s.strip()]
    metrics = run_eval(
        strategies=chosen,
        dataset_path=dataset,
        generate_answers=not no_answers,
    )
    render_comparison_table(metrics, console=console)
    report_dir = get_config().paths.eval_report_dir
    console.print(f"[dim]Reports written to {report_dir}/latest.json and latest.csv[/dim]")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
) -> None:
    """Start the FastAPI Q&A server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "rag.api:app",
        host=host or config.api.host,
        port=port or config.api.port,
        reload=False,
    )


@app.command()
def strategies() -> None:
    """List available retrieval strategies."""
    for name in STRATEGIES:
        console.print(f"- {name}")


@app.command()
def providers() -> None:
    """Show LLM providers and which one is active."""
    settings = get_settings()
    endpoint = build_llm_endpoint(settings)
    table = Table(title="LLM providers (free-first)")
    table.add_column("Provider")
    table.add_column("Env key")
    table.add_column("Default model")
    table.add_column("Ready")
    table.add_row("groq", "GROQ_API_KEY", settings.groq_model, "yes" if settings.groq_api_key else "no")
    table.add_row("huggingface", "HF_TOKEN", settings.hf_model, "yes" if settings.hf_token else "no")
    table.add_row("openai", "OPENAI_API_KEY", settings.openai_model, "yes" if settings.openai_api_key else "no")
    table.add_row("mock", "RAG_MOCK_LLM=1", "mock-llm", "always")
    console.print(table)
    console.print(
        f"Configured LLM_PROVIDER=[bold]{settings.llm_provider or '(auto)'}[/bold] -> "
        f"active [bold]{endpoint.provider}[/bold] ({endpoint.model})"
    )
    console.print(
        "[dim]Get free keys: https://console.groq.com/keys  |  "
        "https://huggingface.co/settings/tokens[/dim]"
    )


if __name__ == "__main__":
    app()
