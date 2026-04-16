"""Report serialisation and formatting for V2 evaluation results.

Provides helpers to persist ``EvalReport`` as JSON and render it as a
human-readable Markdown table (for pasting into documents or CI artefacts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.core.evaluation.pipeline_evaluator import EvalReport

logger = logging.getLogger(__name__)


def save_report_json(report: EvalReport, path: str | Path) -> Path:
    """Persist an evaluation report as pretty-printed JSON.

    Creates parent directories if needed.  Returns the resolved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Evaluation report saved to %s", path)
    return path


def report_to_markdown(report: EvalReport, k_main: int = 5) -> str:
    """Render an ``EvalReport`` as a Markdown string with summary tables."""
    lines: list[str] = []
    lines.append(f"# V2 Retrieval Evaluation Report")
    lines.append("")
    lines.append(f"- **Dataset**: {report.dataset_name}")
    lines.append(f"- **Samples**: {report.dataset_size}")
    lines.append(f"- **Timestamp**: {report.timestamp}")
    lines.append("")

    # ── Config ────────────────────────────────────────────────────────
    lines.append("## Configuration")
    lines.append("")
    for k, v in report.config.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    # ── Stage metrics table ───────────────────────────────────────────
    lines.append("## Stage Metrics")
    lines.append("")
    lines.append(
        f"| Stage | Recall@{k_main} | Precision@{k_main} | F1@{k_main} | MRR | NDCG@{k_main} | Hit@{k_main} | Latency |"
    )
    lines.append("|-------|---------|------------|-------|-----|---------|------|---------|")

    stage_order = [
        "vector_only", "bm25_only", "rrf_fusion",
        "context_assembly", "rerank", "deduplicate",
    ]

    for stage_key in stage_order:
        m = report.stage_metrics.get(stage_key)
        if not m or m.num_samples == 0:
            lines.append(f"| {stage_key} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue

        lines.append(
            f"| {m.stage_name} "
            f"| {m.recall_at_k.get(k_main, 0):.4f} "
            f"| {m.precision_at_k.get(k_main, 0):.4f} "
            f"| {m.f1_at_k.get(k_main, 0):.4f} "
            f"| {m.mrr:.4f} "
            f"| {m.ndcg_at_k.get(k_main, 0):.4f} "
            f"| {m.hit_rate_at_k.get(k_main, 0):.4f} "
            f"| {m.avg_latency_ms:.1f}ms |"
        )

    lines.append("")

    # ── Multi-K detail table ──────────────────────────────────────────
    k_values = report.config.get("k_values", [1, 3, 5, 10])
    lines.append("## Recall@K Across Stages")
    lines.append("")
    k_header = "| Stage | " + " | ".join(f"R@{k}" for k in k_values) + " |"
    k_sep = "|-------| " + " | ".join("----" for _ in k_values) + " |"
    lines.append(k_header)
    lines.append(k_sep)

    for stage_key in stage_order:
        m = report.stage_metrics.get(stage_key)
        if not m or m.num_samples == 0:
            row = f"| {stage_key} | " + " | ".join("N/A" for _ in k_values) + " |"
        else:
            vals = " | ".join(
                f"{m.recall_at_k.get(k, 0):.4f}" for k in k_values
            )
            row = f"| {m.stage_name} | {vals} |"
        lines.append(row)

    lines.append("")

    # ── Generation metrics ────────────────────────────────────────────
    if report.generation_metrics:
        gm = report.generation_metrics
        lines.append("## Generation Metrics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Samples | {gm.num_samples} |")
        lines.append(f"| Avg ROUGE-L | {gm.avg_rouge_l:.4f} |")
        lines.append(f"| Avg Jaccard Similarity | {gm.avg_jaccard:.4f} |")
        lines.append(f"| Avg Citation Accuracy | {gm.avg_citation_accuracy:.4f} |")
        lines.append("")

    return "\n".join(lines)


def save_report_markdown(report: EvalReport, path: str | Path, k_main: int = 5) -> Path:
    """Render and save the report as Markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = report_to_markdown(report, k_main=k_main)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Markdown report saved to %s", path)
    return path
