from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.config import get_config
from rag.eval.metrics import (
    EvalExample,
    StrategyMetrics,
    aggregate,
    answer_token_f1,
    context_precision,
    context_recall,
    faithfulness_proxy,
    hit_at_k,
    load_eval_dataset,
    reciprocal_rank,
)
from rag.pipeline import RAGPipeline
from rag.retrieval import STRATEGIES


def run_eval(
    *,
    strategies: list[str] | None = None,
    dataset_path: str | Path | None = None,
    generate_answers: bool = True,
    top_k: int | None = None,
) -> list[StrategyMetrics]:
    config = get_config()
    pipeline = RAGPipeline(config)
    path = Path(dataset_path or config.paths.eval_dataset)
    examples = load_eval_dataset(path)
    chosen = strategies or list(STRATEGIES)
    k = top_k or config.retrieval.top_k

    results: list[StrategyMetrics] = []
    detailed: dict[str, list[dict]] = {}

    for strategy in chosen:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        rows: list[dict[str, float]] = []
        detail_rows: list[dict] = []
        for example in examples:
            row, detail = _eval_one(
                pipeline,
                example,
                strategy=strategy,
                top_k=k,
                generate_answers=generate_answers,
            )
            rows.append(row)
            detail_rows.append(detail)
        metrics = aggregate(strategy, rows)
        results.append(metrics)
        detailed[strategy] = detail_rows

    _write_reports(results, detailed, Path(config.paths.eval_report_dir))
    return results


def _eval_one(
    pipeline: RAGPipeline,
    example: EvalExample,
    *,
    strategy: str,
    top_k: int,
    generate_answers: bool,
) -> tuple[dict[str, float], dict]:
    retrieved = pipeline.retriever.retrieve(example.question, strategy=strategy, top_k=top_k)  # type: ignore[arg-type]
    answer = ""
    if generate_answers:
        result = pipeline.generator.answer(example.question, retrieved, strategy=strategy)
        answer = result.answer

    row = {
        "hit_at_5": hit_at_k(retrieved, example, k=min(5, top_k)),
        "mrr": reciprocal_rank(retrieved, example),
        "context_recall": context_recall(retrieved, example),
        "context_precision": context_precision(retrieved, example),
        "answer_f1": answer_token_f1(answer, example.reference_answer) if generate_answers else 0.0,
        "faithfulness": faithfulness_proxy(answer, retrieved) if generate_answers else 0.0,
    }
    detail = {
        "id": example.id,
        "question": example.question,
        "strategy": strategy,
        "metrics": row,
        "retrieved_sources": [r.chunk.source for r in retrieved],
        "answer": answer,
    }
    return row, detail


def render_comparison_table(metrics: list[StrategyMetrics], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="RAG Retrieval Strategy Comparison", show_lines=False)
    table.add_column("Strategy", style="bold")
    table.add_column("Hit@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Ctx Recall", justify="right")
    table.add_column("Ctx Prec", justify="right")
    table.add_column("Ans F1", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("N", justify="right")

    best_hit = max((m.hit_at_5 for m in metrics), default=0.0)
    for m in metrics:
        marker = " *" if m.hit_at_5 == best_hit and best_hit > 0 else ""
        table.add_row(
            f"{m.strategy}{marker}",
            f"{m.hit_at_5:.3f}",
            f"{m.mrr:.3f}",
            f"{m.context_recall:.3f}",
            f"{m.context_precision:.3f}",
            f"{m.answer_f1:.3f}",
            f"{m.faithfulness:.3f}",
            str(m.n),
        )
    console.print(table)
    console.print("[dim]* = best Hit@5[/dim]")


def _write_reports(metrics: list[StrategyMetrics], detailed: dict, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"eval_{stamp}.json"
    csv_path = report_dir / f"eval_{stamp}.csv"
    latest_json = report_dir / "latest.json"
    latest_csv = report_dir / "latest.csv"

    payload = {
        "created_at": stamp,
        "summary": [m.to_dict() for m in metrics],
        "details": detailed,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "strategy",
        "n",
        "hit_at_5",
        "mrr",
        "context_recall",
        "context_precision",
        "answer_f1",
        "faithfulness",
    ]
    for path in (csv_path, latest_csv):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics:
                writer.writerow(m.to_dict())
    return json_path
