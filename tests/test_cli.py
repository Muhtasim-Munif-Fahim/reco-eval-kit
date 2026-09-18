from reco_eval_kit.cli import main


BEYOND_ACCURACY_METRICS = (
    "CatalogCoverage",
    "NoveltyBits",
    "IntraListDiversity",
    "Unexpectedness",
    "ProfileUnexpectedness",
    "Serendipity",
    "ProfileSerendipity",
)


def test_cli_report_includes_beyond_accuracy_metrics(tmp_path):
    output = tmp_path / "report.md"
    assert (
        main(
            [
                "--n-users",
                "8",
                "--n-items",
                "12",
                "--n-interactions",
                "40",
                "--k",
                "3",
                "--seed",
                "0",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    text = output.read_text(encoding="utf-8")
    for metric in BEYOND_ACCURACY_METRICS:
        assert f"| {metric} |" in text
