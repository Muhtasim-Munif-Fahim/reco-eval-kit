from pathlib import Path

from reco_eval_kit.report import render_report, save_report


def test_render_includes_model_sections_and_metrics(tmp_path):
    text = render_report(
        {"popularity": {"NDCG@K": 0.123456, "MRR": 0.5}},
        title="Demo Run",
    )
    assert "# Demo Run" in text
    assert "## popularity" in text
    assert "| NDCG@K | 0.1235 |" in text


def test_render_includes_dataset_summary_table():
    text = render_report({"m": {"MRR": 1.0}}, dataset_summary={"users": 5, "k": 10})
    assert "## Dataset" in text
    assert "| users | 5 |" in text
    assert "| k | 10 |" in text


def test_save_report_writes_file_and_creates_parents(tmp_path):
    target = tmp_path / "nested" / "dir" / "report.md"
    saved = save_report("# content", target)
    assert saved == Path(target)
    assert saved.read_text(encoding="utf-8") == "# content"


def test_values_are_formatted_to_four_decimals():
    text = render_report({"m": {"HitRate@K": 1}})
    assert "| HitRate@K | 1.0000 |" in text