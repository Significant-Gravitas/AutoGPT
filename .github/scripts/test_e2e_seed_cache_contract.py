import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

SEED_KEY_STEP = "Set up tests - Compute E2E test data cache key"
SEED_PULL_STEP = "Set up tests - Pull cached E2E test data from GHCR"
SEED_RESTORE_STEP = "Set up tests - Load cached E2E test data"
SEED_CREATE_STEP = "Set up tests - Create E2E test data"
SEED_VERIFY_STEP = "Set up tests - Verify seeded credentials with fresh encryption key"
SEED_SAVE_STEP = "Push E2E test data to GHCR"
PLAYWRIGHT_STEP = "Run Playwright E2E suite"
PLAYWRIGHT_REPORT_STEP = "Validate Playwright JSON report"
# What the seed prints once every item has an embedding.
EMBEDDINGS_COMPLETE = "Embeddings complete:"
# Backend modules whose functions the seed calls to write rows into the dump.
# The dump's encrypted payloads are re-encrypted on every run, so the
# encryption module is not one of them.
SEED_WRITERS = (
    "api/features/library/db",
    "api/features/library/model",
    "api/features/orgs/db",
    "api/features/search/embeddings",
    "api/features/store/db",
    "data/auth/api_key",
    "data/credit",
    "data/graph",
    "data/org_migration",
    "data/profile",
    "data/user",
    "util/json",
)


def _step(workflow: str, name: str) -> str:
    return workflow.split(f"- name: {name}\n", 1)[1].split("\n      - name:", 1)[0]


def _matrix_leg(workflow: str, suite: str) -> dict[str, str]:
    e2e_job = workflow.split("\n  e2e_test:\n", 1)[1].split("\n    steps:\n", 1)[0]
    leg = e2e_job.split(f"- suite: {suite}\n", 1)[1].split("- suite:", 1)[0]
    return dict(re.findall(r"^\s+(\w+): (\S+)$", leg, re.MULTILINE))


class E2EMatrixContractTests(unittest.TestCase):
    def test_legs_keep_separate_seed_caches(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("matrix.embeddings", _step(workflow, SEED_KEY_STEP))
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
        # The trailing newline matters: the key step's sed names the create
        # step too, earlier in the file.
        self.assertLess(
            workflow.index(f"- name: {SEED_CREATE_STEP}\n"),
            workflow.index(f"- name: {name}\n"),
        )
        self.assertLess(
            workflow.index(f"- name: {name}\n"),
            workflow.index(f"- name: {PLAYWRIGHT_STEP}\n"),
        )

    def test_seed_dump_excludes_migration_owned_activation_policy(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        dump = workflow.split("pg_dump -U postgres", 1)[1].split(
            "postgres > /tmp/e2e_test_data.sql", 1
        )[0]
        self.assertIn("--exclude-table='platform.\"PaidUsageActivationPolicy\"'", dump)

    def test_seed_cache_key_is_computed_from_pristine_checkout(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        e2e_job = workflow.split("\n  e2e_test:\n", 1)[1]
        steps = re.findall(r"^      - name: (.+)$", e2e_job, re.MULTILINE)
        self.assertEqual(
            steps[:2],
            ["Checkout repository", SEED_KEY_STEP],
            "seed cache key must be evaluated before generated env/Compose files exist",
        )

    def test_seed_dump_is_not_in_the_actions_cache(self):
        # The repo's Actions cache sits at its cap and evicted the dump within
        # minutes: 9 hits in 206 smoke jobs (SECRT-2699).
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        for step in workflow.split("\n      - name: ")[1:]:
            if "uses: actions/cache" in step:
                with self.subTest(step=step.splitlines()[0]):
                    self.assertNotIn("e2e_test_data", step)

    def test_seed_cache_key_covers_only_seed_inputs(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        key_step = _step(workflow, SEED_KEY_STEP)
        script = key_step.split("run: |\n", 1)[1]
        hashed = set(re.findall(r"^\s+(autogpt_platform/\S+?)(?: \\)?$", script, re.M))
        self.assertEqual(
            hashed,
            {
                "autogpt_platform/backend/test/e2e_test_data.py",
                "autogpt_platform/backend/agents/calculator-agent.json",
                "autogpt_platform/backend/schema.prisma",
                "autogpt_platform/backend/migrations",
                # Backend helpers the seed calls that write the dumped rows.
                *(
                    f"autogpt_platform/backend/backend/{module}.py"
                    for module in SEED_WRITERS
                ),
            },
        )
        # The dump step's own text, not the whole workflow file.
        self.assertIn(f"/- name: {SEED_CREATE_STEP}$/", script)

        root = Path(__file__).resolve().parents[2]
        # git ls-files skips a path that is gone, so a rename would quietly
        # drop it from the key.
        for path in hashed:
            with self.subTest(path=path):
                self.assertTrue((root / path).exists())
        outputs = []
        for _ in range(2):
            with tempfile.NamedTemporaryFile("r", suffix=".out") as out:
                subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    cwd=root,
                    env={**os.environ, "EMBEDDINGS": "emb", "GITHUB_OUTPUT": out.name},
                    check=True,
                )
                outputs.append(out.read())
        self.assertEqual(outputs[0], outputs[1], "seed cache key is not deterministic")
        tag = re.fullmatch(r"tag=(\S+)\n", outputs[0])
        self.assertIsNotNone(tag, outputs[0])
        assert tag
        # A valid OCI tag that names the embeddings mode.
        self.assertRegex(tag[1], r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
        self.assertIn("-emb-", tag[1])

    def test_seed_dump_restores_from_ghcr_or_falls_back_to_seeding(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("E2E_SEED_IMAGE: ghcr.io/significant-gravitas/", workflow)
        pull = _step(workflow, SEED_PULL_STEP)
        restore = _step(workflow, SEED_RESTORE_STEP)
        create = _step(workflow, SEED_CREATE_STEP)
        # An unreachable or missing image never fails the job.
        self.assertIn("continue-on-error: true", pull)
        self.assertIn("timeout-minutes:", pull)
        self.assertIn("steps.e2e-seed-pull.outputs.hit == 'true'", restore)
        # A dump that no longer loads rolls back whole and is seeded fresh.
        self.assertIn("--single-transaction", restore)
        self.assertIn("ON_ERROR_STOP=1", restore)
        self.assertLess(
            restore.index("--single-transaction"), restore.index("restored=true")
        )
        self.assertEqual(
            re.findall(r"^\s+if: (.+)$", create, re.M),
            ["matrix.seed && steps.e2e-seed-restore.outputs.restored != 'true'"],
        )
        self.assertLess(
            workflow.index(f"- name: {SEED_KEY_STEP}\n"),
            workflow.index(
                "- name: Set up Platform - Log in to the GHCR build cache\n"
            ),
        )
        self.assertLess(
            workflow.index(
                "- name: Set up Platform - Log in to the GHCR build cache\n"
            ),
            workflow.index(f"- name: {SEED_PULL_STEP}\n"),
        )

    def test_seed_dump_is_saved_only_where_build_cache_is_written(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        job_env = workflow.split("\n  e2e_test:\n", 1)[1].split("\n    steps:\n", 1)[0]
        self.assertIn("WRITE_BUILD_CACHE: ${{", job_env)
        save = _step(workflow, SEED_SAVE_STEP)
        self.assertEqual(
            re.findall(r"^\s+if: (.+)$", save, re.M),
            [
                "matrix.seed && env.WRITE_BUILD_CACHE == 'true' && "
                "steps.e2e-seed-create.outcome == 'success' && "
                "steps.e2e-seed-create.outputs.publishable == 'true' && "
                "steps.e2e-seed-verify.outcome == 'success' && "
                "steps.e2e-playwright.outcome == 'success' && "
                "steps.e2e-playwright-report.outcome == 'success'"
            ],
        )
        self.assertIn("continue-on-error: true", save)
        self.assertIn('docker push "$ref"', save)
        self.assertIn("id: e2e-seed-create", _step(workflow, SEED_CREATE_STEP))
        self.assertIn("id: e2e-seed-verify", _step(workflow, SEED_VERIFY_STEP))
        self.assertIn("id: e2e-playwright", _step(workflow, PLAYWRIGHT_STEP))
        self.assertIn(
            "id: e2e-playwright-report", _step(workflow, PLAYWRIGHT_REPORT_STEP)
        )
        # The seed swallows graph, store and fixture insert failures, so only
        # a dump the whole suite passed on is published; a broken one would
        # otherwise fail every run that restores it.
        self.assertLess(
            workflow.index(f"- name: {PLAYWRIGHT_REPORT_STEP}\n"),
            workflow.index(f"- name: {SEED_SAVE_STEP}\n"),
        )

    def test_seed_dump_with_incomplete_embeddings_is_not_published(self):
        # The seed warns and exits 0 when the embedding backfill errors, stalls
        # or runs out of time; a dump without full coverage must not be reused.
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        seed = (
            Path(__file__).resolve().parents[2]
            / "autogpt_platform/backend/test/e2e_test_data.py"
        ).read_text(encoding="utf-8")
        create = _step(workflow, SEED_CREATE_STEP)
        self.assertIn(EMBEDDINGS_COMPLETE, seed)
        self.assertIn(f"grep -q '{EMBEDDINGS_COMPLETE}' /tmp/e2e_seed.log", create)
        self.assertIn("matrix.embeddings", create)
        self.assertIn('echo "publishable=true" >> "$GITHUB_OUTPUT"', create)
        self.assertIn(
            "steps.e2e-seed-create.outputs.publishable == 'true'",
            _step(workflow, SEED_SAVE_STEP),
        )

    def test_seed_dump_is_pushed_and_pulled_under_one_name(self):
        # A one-sided rename would make every pull a miss while CI stays green.
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        dumped = re.findall(r"postgres > /tmp/(\S+)$", workflow, re.M)
        pushed = re.findall(r"tar -C /tmp -c (\S+) \|", _step(workflow, SEED_SAVE_STEP))
        pulled = re.findall(
            r'docker cp "\$container:/(\S+)" /tmp/(\S+)$',
            _step(workflow, SEED_PULL_STEP),
            re.M,
        )
        restored = re.findall(
            r"cat /tmp/(\S+)$", _step(workflow, SEED_RESTORE_STEP), re.M
        )
        self.assertEqual(dumped, ["e2e_test_data.sql"])
        self.assertEqual(pushed, dumped)
        self.assertEqual(pulled, [(dumped[0], dumped[0])])
        self.assertEqual(restored, dumped)


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
