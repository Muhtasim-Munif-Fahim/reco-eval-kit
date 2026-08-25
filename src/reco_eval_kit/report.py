"""Markdown rendering of evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union


def render_report(
    model_results: Mapping[str, Mapping[str, float]],
    dataset_summary: Union[Mapping[str, object], None] = None,
    title: str = "Recommender Evaluation Report",
) -> str:
    """Render per-model metric dictionaries as a markdown document.

    ``model_results`` maps model name to a mapping of metric name to float.
    ``dataset_summary`` optionally adds a key/value table describing the
    evaluation setup above the results.
    """
    lines = [f"# {title}", ""]
    if dataset_summary:
        lines += ["## Dataset", "", "| Property | Value |", "| --- | --- |"]
        for key, value in dataset_summary.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    for model_name, metrics in model_results.items():
        lines += [f"## {model_name}", "", "| Metric | Value |", "| --- | --- |"]
        for metric_name, value in metrics.items():
            lines.append(f"| {metric_name} | {float(value):.4f} |")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def save_report(markdown: str, path) -> Path:
    """Write the markdown document, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path