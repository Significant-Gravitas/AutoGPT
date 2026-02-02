from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_runledger_google_docs_format_text(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[2]
    suite_dir = backend_root / "test" / "runledger"
    output_dir = tmp_path / "runledger_out"

    cmd = [
        sys.executable,
        "-m",
        "runledger",
        "run",
        str(suite_dir),
        "--mode",
        "live",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=backend_root)
