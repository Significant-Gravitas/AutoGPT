import fnmatch
import re
import unittest
from pathlib import Path


def _matrix_leg(workflow: str, suite: str) -> dict[str, str]:
    e2e_job = workflow.split("\n  e2e_test:\n", 1)[1].split("\n    steps:\n", 1)[0]
    leg = e2e_job.split(f"- suite: {suite}\n", 1)[1].split("- suite:", 1)[0]
    return dict(re.findall(r"^\s+(\w+): (\S+)$", leg, re.MULTILINE))


class E2EMatrixContractTests(unittest.TestCase):
    def test_legs_keep_separate_seed_caches(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        key = next(
            line for line in workflow.splitlines() if "key: e2e-test-data-" in line
        )
        self.assertIn("matrix.embeddings", key)
        self.assertEqual(_matrix_leg(workflow, "smoke")["embeddings"], "true")
        self.assertEqual(_matrix_leg(workflow, "journeys")["embeddings"], "false")

    def test_only_smoke_writes_shared_caches(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        write_cache = next(
            line for line in workflow.splitlines() if "WRITE_BUILD_CACHE: ${{" in line
        )
        self.assertIn("matrix.suite == 'smoke' &&", write_cache)
        for step in (
            "Export Next.js build cache from BuildKit",
            "Save Next.js build cache",
            "Set up tests - Cache Playwright browsers",
        ):
            with self.subTest(step=step):
                body = workflow.split(f"- name: {step}\n", 1)[1].split(
                    "\n      - name:", 1
                )[0]
                self.assertIn("matrix.suite == 'smoke'", body)

    def test_only_journeys_answer_copilot_from_scripts(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertEqual(_matrix_leg(workflow, "smoke")["scripted_model"], "false")
        self.assertEqual(_matrix_leg(workflow, "journeys")["scripted_model"], "true")
        prepare = workflow.split("- name: Set up Platform - Prepare Docker build\n", 1)[
            1
        ].split("\n      - name:", 1)[0]
        self.assertIn("-f docker-compose.journeys.yml", prepare)
        self.assertIn("matrix.scripted_model", prepare)

    def test_journeys_check_is_advisory(self):
        checker = (
            Path(__file__).resolve().parents[1]
            / "workflows/scripts/check_actions_status.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"end-to-end tests (journeys)"', checker)


class E2ESeedCacheContractTests(unittest.TestCase):
    def test_fresh_and_cached_seed_credentials_are_verified_before_tests(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        name = "Set up tests - Verify seeded credentials with fresh encryption key"
        verification = workflow.split(f"- name: {name}\n", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn("--refresh-credentials-only", verification)
        # Runs whenever the leg seeds, cached dump or fresh; only the
        # journeys leg, which seeds nothing, skips it.
        self.assertEqual(
            re.findall(r"^\s+if: (.+)$", verification, re.M), ["matrix.seed"]
        )
        self.assertEqual(_matrix_leg(workflow, "smoke")["seed"], "true")
        self.assertNotIn("continue-on-error", verification)
        self.assertLess(
            workflow.index("- name: Set up tests - Create E2E test data"),
            workflow.index(f"- name: {name}"),
        )
        self.assertLess(
            workflow.index(f"- name: {name}"),
            workflow.index("- name: Run Playwright E2E suite"),
        )

    def test_seed_cache_key_uses_pristine_checkout(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        e2e_job = workflow.split("\n  e2e_test:\n", 1)[1]
        steps = re.findall(r"^      - name: (.+)$", e2e_job, re.MULTILINE)
        self.assertEqual(
            steps[:2],
            ["Checkout repository", "Set up tests - Cache E2E test data"],
            "seed cache key must be evaluated before generated env/Compose files exist",
        )

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


class E2EImageBuildContractTests(unittest.TestCase):
    def test_backend_images_reinstall_os_packages_daily(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        build = workflow.split(
            "- name: Set up Platform - Build backend Docker images\n", 1
        )[1].split("\n      - name:", 1)[0]
        # APT_REFRESH in backend/Dockerfile: the date re-runs the package
        # installs once a day, so E2E tests the OS packages that ship.
        self.assertIn('--set "*.args.APT_REFRESH=$(date -u +%F)"', build)


class E2ECoverageContractTests(unittest.TestCase):
    def test_test_floor_includes_shared_file_security_suite(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("--min-tests ${{ matrix.min_tests }}", workflow)
        self.assertEqual(_matrix_leg(workflow, "smoke")["min_tests"], "28")
        self.assertEqual(_matrix_leg(workflow, "journeys")["min_tests"], "11")
        # Smoke stays strict: no test.fail() known failures.
        self.assertEqual(_matrix_leg(workflow, "smoke")["known_failures"], "false")

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
