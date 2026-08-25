"""Run the bundled end-to-end demo and write examples/output/demo_report.md."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reco_eval_kit.cli import main  # noqa: E402


if __name__ == "__main__":
    output = ROOT / "examples" / "output" / "demo_report.md"
    raise SystemExit(
        main(
            [
                "--n-users", "60",
                "--n-items", "150",
                "--n-interactions", "2500",
                "--seed", "7",
                "--k", "10",
                "--output", str(output),
            ]
        )
    )