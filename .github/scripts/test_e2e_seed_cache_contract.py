import fnmatch
import re
import unittest
from pathlib import Path


class E2ESeedCacheContractTests(unittest.TestCase):
    def test_seed_cache_covers_transitive_inputs(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        key = next(
            line for line in workflow.splitlines() if "key: e2e-test-data-" in line
        )
        patterns = re.findall(r"'([^']+)'", key)
        for path in (
            "autogpt_platform/backend/test/e2e_test_data.py",
            "autogpt_platform/backend/backend/data/graph.py",
            "autogpt_platform/backend/backend/blocks/calculator.py",
            "autogpt_platform/backend/agents/calculator-agent.json",
            "autogpt_platform/backend/schema.prisma",
            "autogpt_platform/backend/poetry.lock",
            "autogpt_platform/autogpt_libs/autogpt_libs/auth/models.py",
            "autogpt_platform/docker-compose.yml",
            "autogpt_platform/.env.default",
            ".github/workflows/platform-fullstack-ci.yml",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns),
                    f"seed cache key does not cover {path}",
                )


class E2ECoverageContractTests(unittest.TestCase):
    def test_codecov_requires_validated_coverage(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        validator = workflow.split("- name: Validate coverage report\n", 1)[1].split(
            "\n      - name:", 1
        )[0]
        upload = workflow.split("- name: Upload E2E coverage to Codecov\n", 1)[1]
        self.assertIn("id: validate-e2e-coverage", validator)
        self.assertIn("if: ${{ always() }}", validator)
        self.assertNotIn("continue-on-error", validator)
        self.assertIn(
            "if: ${{ !cancelled() && steps.validate-e2e-coverage.outcome == 'success' }}",
            upload,
        )


if __name__ == "__main__":
    unittest.main()
