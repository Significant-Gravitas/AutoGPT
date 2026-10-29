from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "platform-single-container-docker.yml"
)
IGNORE_FILE = "autogpt_platform/single-container/.trivyignore.yaml"


class ScanIgnoreTest(unittest.TestCase):
    def test_every_trivy_scan_uses_the_shared_ignore_file(self) -> None:
        # The CI build and the release digest are scanned by separate steps; an
        # entry that only one of them honours fails the other on a false positive.
        steps = WORKFLOW_PATH.read_text(encoding="utf-8").split("- name: ")[1:]
        scans = [step for step in steps if "aquasecurity/trivy-action@" in step]
        self.assertGreaterEqual(len(scans), 4)
        for step in scans:
            self.assertIn(f"trivyignores: {IGNORE_FILE}", step, step.splitlines()[0])
        self.assertTrue((REPO_ROOT / IGNORE_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
