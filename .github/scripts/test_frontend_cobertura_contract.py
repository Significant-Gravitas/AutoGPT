import unittest
from pathlib import Path

import yaml


class FrontendCoverageContractTests(unittest.TestCase):
    def test_coverage_waits_for_every_integration_shard(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load(
            (root / ".github/workflows/platform-frontend-ci.yml").read_text()
        )
        matrix = workflow["jobs"]["integration_test"]["strategy"]["matrix"]
        codecov = yaml.safe_load((root / "codecov.yml").read_text())
        self.assertEqual(
            codecov["flags"]["platform-frontend"]["after_n_builds"],
            len(matrix["shard"]),
        )

    def test_codecov_requires_validated_coverage(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-frontend-ci.yml"
        ).read_text(encoding="utf-8")
        validator = workflow.split("- name: Validate coverage report\n", 1)[1].split(
            "\n      - name:", 1
        )[0]
        upload = workflow.split("- name: Upload coverage reports to Codecov\n", 1)[1]
        self.assertIn("id: validate-frontend-coverage", validator)
        self.assertIn("if: ${{ always() }}", validator)
        self.assertNotIn("continue-on-error", validator)
        self.assertIn(
            "if: ${{ !cancelled() && steps.validate-frontend-coverage.outcome == 'success' }}",
            upload,
        )


if __name__ == "__main__":
    unittest.main()
