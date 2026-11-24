import ast
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

SEED_KEY_STEP = "Set up tests - Compute E2E test data cache key"
SEED_PULL_STEP = "Set up tests - Pull cached E2E test data from GHCR"
SEED_RESTORE_STEP = "Set up tests - Load cached E2E test data"
SEED_CREATE_STEP = "Set up tests - Create E2E test data"
SEED_VERIFY_STEP = "Set up tests - Verify seeded credentials with fresh encryption key"
SEED_VERIFY_CACHED_STEP = (
    "Set up tests - Verify cached seed credentials with fresh encryption key"
)
FRONTEND_WAIT_STEP = "Wait for frontend image build"
SERVICES_START_STEP = "Set up Platform - Start (all other services)"
STORE_AGENTS_STEP = "Set up tests - Load the official marketplace agents"
SEED_SAVE_STEP = "Push E2E test data to GHCR"
PLAYWRIGHT_STEP = "Run Playwright E2E suite"
PLAYWRIGHT_REPORT_STEP = "Validate Playwright JSON report"
INVENTORY_STEP = "Set up tests - Count the shard's tests"
STACK_WAIT_STEP = "Wait for backend image build and database"
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


def _matrix_legs(workflow: str, suite: str) -> list[dict[str, str]]:
    """Every e2e matrix leg of a suite, as its `key: value` lines."""
    e2e_job = workflow.split("\n  e2e_test:\n", 1)[1].split("\n    steps:\n", 1)[0]
    legs: list[dict[str, str]] = []
    for line in e2e_job.split("\n        include:\n", 1)[1].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" " * 10):
            break
        entry = re.fullmatch(r"\s+(- )?(\w+): (.+?)\s*", line)
        assert entry, line
        if entry[1]:
            legs.append({})
        legs[-1][entry[2]] = entry[3]
    return [leg for leg in legs if leg["suite"] == suite]


def _matrix_leg(workflow: str, suite: str) -> dict[str, str]:
    legs = _matrix_legs(workflow, suite)
    assert (
        len(legs) == 1
    ), f"expected one {suite} leg in the e2e matrix, found {len(legs)}"
    return legs[0]


def _render(template: str, leg: dict[str, str]) -> str:
    """Expands the e2e job's `${{ matrix.* }}` and shard expressions for a leg."""

    def expand(expression: re.Match[str]) -> str:
        if plain := re.fullmatch(r"matrix\.(\w+)", expression[1]):
            return leg.get(plain[1], "")
        sharded = re.fullmatch(
            r"matrix\.shard_index && format\('([^']*)'((?:, matrix\.\w+)+)\) \|\| ''",
            expression[1],
        )
        assert sharded, expression[1]
        if "shard_index" not in leg:
            return ""
        keys = re.findall(r"matrix\.(\w+)", sharded[2])
        return sharded[1].format(*(leg[key] for key in keys))

    return re.sub(r"\$\{\{ (.+?) \}\}", expand, template)


def _shard_expressions(text: str) -> list[str]:
    return [
        expression
        for expression in re.findall(r"\$\{\{ .+? \}\}", text)
        if "matrix.shard_index &&" in expression
    ]


def _e2e_step_names(workflow: str) -> list[str]:
    e2e_job = workflow.split("\n  e2e_test:\n", 1)[1]
    return re.findall(r"^      - name: (.+)$", e2e_job, re.MULTILINE)


class E2EMatrixContractTests(unittest.TestCase):
    def test_legs_keep_separate_seed_caches(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("matrix.embeddings", _step(workflow, SEED_KEY_STEP))
        self.assertEqual(_matrix_leg(workflow, "smoke")["embeddings"], "true")
        self.assertEqual(
            {leg["embeddings"] for leg in _matrix_legs(workflow, "journeys")},
            {"false"},
        )

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
        self.assertEqual(
            {leg["scripted_model"] for leg in _matrix_legs(workflow, "journeys")},
            {"true"},
        )
        prepare = workflow.split("- name: Set up Platform - Prepare Docker build\n", 1)[
            1
        ].split("\n      - name:", 1)[0]
        self.assertIn("-f docker-compose.journeys.yml", prepare)
        self.assertIn("matrix.scripted_model", prepare)

    def test_journeys_load_the_roster_workflows_before_the_roster_publishes(self):
        # publish_skills resolves each roster workflow to an official listing
        # when it starts, and skips any it can't find: loaded after it, a hired
        # expert gets no workflows. Beside the Next.js build it could also run
        # the runner out of memory, like the seed.
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertEqual(_matrix_leg(workflow, "smoke")["store_agents"], "false")
        self.assertEqual(
            {leg["store_agents"] for leg in _matrix_legs(workflow, "journeys")},
            {"true"},
        )
        load = _step(workflow, STORE_AGENTS_STEP)
        # The whole condition: a widened one, such as `|| matrix.suite ==
        # 'smoke'`, would load the listings on smoke too.
        self.assertEqual(
            [
                line.strip()
                for line in load.splitlines()
                if line.strip().startswith("if:")
            ],
            ["if: matrix.store_agents"],
        )
        # With images the stack serves itself: the CSV's production images on
        # GCS had the image optimizer fetch from outside the stack (#17298).
        self.assertIn("python -m test.load_store_agents --local-media", load)
        steps = _e2e_step_names(workflow)
        self.assertLess(steps.index(FRONTEND_WAIT_STEP), steps.index(STORE_AGENTS_STEP))
        self.assertLess(
            steps.index(STORE_AGENTS_STEP), steps.index(SERVICES_START_STEP)
        )

    def test_journeys_run_as_identical_shards(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        legs = _matrix_legs(workflow, "journeys")
        self.assertEqual([leg.get("shard_index") for leg in legs], ["1", "2"])
        self.assertEqual({leg.get("shard_total") for leg in legs}, {str(len(legs))})
        # Everything else, the pinned total included, is the same on each.
        first = {key: value for key, value in legs[0].items() if key != "shard_index"}
        for leg in legs[1:]:
            with self.subTest(shard=leg["shard_index"]):
                self.assertEqual(
                    {key: value for key, value in leg.items() if key != "shard_index"},
                    first,
                )
        self.assertNotIn("shard_index", _matrix_leg(workflow, "smoke"))

        # Each leg hands Playwright its own shard; smoke runs whole.
        (shard_flag,) = _shard_expressions(_step(workflow, PLAYWRIGHT_STEP))
        self.assertEqual(
            [_render(shard_flag, leg) for leg in legs], ["--shard=1/2", "--shard=2/2"]
        )
        self.assertEqual(_render(shard_flag, _matrix_leg(workflow, "smoke")), "")

    def test_each_leg_has_its_own_job_and_artifact_names(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        e2e_job = workflow.split("\n  e2e_test:\n", 1)[1]
        name = re.search(r"^    name: (.+)$", e2e_job, re.M)
        leg_name = re.search(r"^      E2E_LEG: (.+)$", e2e_job, re.M)
        assert name and leg_name
        legs = [_matrix_leg(workflow, "smoke"), *_matrix_legs(workflow, "journeys")]
        self.assertEqual(
            [_render(name[1], leg) for leg in legs],
            [
                "end-to-end tests (smoke)",
                "end-to-end tests (journeys 1/2)",
                "end-to-end tests (journeys 2/2)",
            ],
        )
        self.assertEqual(
            [_render(leg_name[1], leg) for leg in legs],
            ["smoke", "journeys-1", "journeys-2"],
        )
        # Two uploads of one artifact name in a run fail the second.
        for step, artifact in (
            ("Upload Playwright report", "playwright-report-${{ env.E2E_LEG }}"),
            (
                "Upload Playwright test results",
                "playwright-test-results-${{ env.E2E_LEG }}",
            ),
        ):
            with self.subTest(step=step):
                self.assertIn(f"name: {artifact}\n", _step(workflow, step))

    def test_every_journeys_shard_is_advisory(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        checker = (
            Path(__file__).resolve().parents[1]
            / "workflows/scripts/check_actions_status.py"
        ).read_text(encoding="utf-8")

        def literal(target: str):
            return next(
                ast.literal_eval(node.value)
                for node in ast.parse(checker).body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(each, ast.Name) and each.id == target
                    for each in node.targets
                )
            )

        names = literal("ADVISORY_CHECK_RUNS")
        prefixes = literal("ADVISORY_CHECK_RUN_PREFIXES")

        # check_actions_status.is_advisory, without importing its requests.
        def is_advisory(check: str) -> bool:
            return check in names or check.startswith(prefixes)

        name = re.search(
            r"^    name: (.+)$", workflow.split("\n  e2e_test:\n", 1)[1], re.M
        )
        assert name
        for leg in _matrix_legs(workflow, "journeys"):
            with self.subTest(leg=_render(name[1], leg)):
                self.assertTrue(is_advisory(_render(name[1], leg)))
        self.assertFalse(is_advisory(_render(name[1], _matrix_leg(workflow, "smoke"))))
        # A shard added later is advisory with no checker edit.
        self.assertTrue(is_advisory("end-to-end tests (journeys 3/3)"))


class E2ESeedCacheContractTests(unittest.TestCase):
    def test_fresh_and_cached_seed_credentials_are_verified_before_tests(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        cached = _step(workflow, SEED_VERIFY_CACHED_STEP)
        fresh = _step(workflow, SEED_VERIFY_STEP)
        # Exactly one of the two runs whenever the leg seeds: the cached check
        # after a restore, the fresh one whenever the create step runs.
        self.assertEqual(
            re.findall(r"^\s+if: (.+)$", cached, re.MULTILINE),
            ["steps.e2e-seed-restore.outputs.restored == 'true'"],
        )
        self.assertEqual(
            re.findall(r"^\s+if: (.+)$", fresh, re.MULTILINE),
            re.findall(
                r"^\s+if: (.+)$", _step(workflow, SEED_CREATE_STEP), re.MULTILINE
            ),
        )
        self.assertEqual(_matrix_leg(workflow, "smoke")["seed"], "true")
        for verification in (cached, fresh):
            with self.subTest(step=verification.splitlines()[0]):
                self.assertIn("--refresh-credentials-only", verification)
                self.assertNotIn("continue-on-error", verification)
        steps = _e2e_step_names(workflow)
        self.assertLess(
            steps.index(SEED_RESTORE_STEP), steps.index(SEED_VERIFY_CACHED_STEP)
        )
        self.assertLess(steps.index(SEED_CREATE_STEP), steps.index(SEED_VERIFY_STEP))
        for name in (SEED_VERIFY_CACHED_STEP, SEED_VERIFY_STEP):
            self.assertLess(steps.index(name), steps.index(PLAYWRIGHT_STEP))

    def test_services_and_seed_wait_for_the_frontend_build_and_seed_is_time_boxed(
        self,
    ):
        # Beside the Next.js build, seeding froze the runner for up to 48 minutes
        # (OPEN-3734) and the started backend services lost it (SECRT-3254).
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        seed = (
            Path(__file__).resolve().parents[2]
            / "autogpt_platform/backend/test/e2e_test_data.py"
        ).read_text(encoding="utf-8")
        steps = _e2e_step_names(workflow)
        self.assertIn(
            "wait: [platform-build-frontend]", _step(workflow, FRONTEND_WAIT_STEP)
        )
        for name in (SERVICES_START_STEP, SEED_VERIFY_CACHED_STEP, SEED_CREATE_STEP):
            with self.subTest(step=name):
                self.assertLess(steps.index(FRONTEND_WAIT_STEP), steps.index(name))

        timeout = re.search(
            r"^\s+timeout-minutes: (\d+)$",
            _step(workflow, SEED_CREATE_STEP),
            re.MULTILINE,
        )
        deadline = re.search(r'"E2E_EMBEDDING_TIMEOUT_SECONDS", "(\d+)"', seed)
        assert timeout and deadline
        # The seed's own embedding deadline must give up first, with time left to
        # dump, so a slow seed still runs the suite instead of being killed.
        self.assertGreater(int(timeout[1]) * 60, int(deadline[1]) + 120)

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
    def test_backend_images_reinstall_os_packages_when_debian_publishes(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        build = workflow.split(
            "- name: Set up Platform - Build backend Docker images\n", 1
        )[1].split("\n      - name:", 1)[0]
        # APT_REFRESH in backend/Dockerfile: the key changes daily and when
        # Debian publishes security updates, re-running the package installs,
        # so E2E tests the OS packages that ship.
        key = 'apt_refresh="$(python3 "$GITHUB_WORKSPACE/.github/scripts/apt_refresh.py")"'
        bake = '--set "*.args.APT_REFRESH=$apt_refresh"'
        self.assertIn(key, build)
        self.assertIn(bake, build)
        self.assertLess(build.index(key), build.index(bake))


class E2ECoverageContractTests(unittest.TestCase):
    def test_test_floor_includes_shared_file_security_suite(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        self.assertEqual(_matrix_leg(workflow, "smoke")["min_tests"], "37")
        # The whole journeys suite's inventory, one total on every shard.
        self.assertEqual(
            {leg["min_tests"] for leg in _matrix_legs(workflow, "journeys")}, {"81"}
        )
        # Smoke stays strict: no test.fail() known failures.
        self.assertEqual(_matrix_leg(workflow, "smoke")["known_failures"], "false")

    def test_a_shard_is_held_to_its_listed_count_and_the_suite_to_its_pin(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        inventory = _step(workflow, INVENTORY_STEP)
        report = _step(workflow, PLAYWRIGHT_REPORT_STEP)
        run = _step(workflow, PLAYWRIGHT_STEP)
        self.assertIn("id: e2e-inventory\n", inventory)
        self.assertIn("if: matrix.shard_index\n", inventory)
        # A shard's floor is its own count; smoke keeps its pin.
        self.assertIn(
            "MIN_TESTS: ${{ matrix.shard_index && steps.e2e-inventory.outputs.tests"
            " || matrix.min_tests }}\n",
            report,
        )
        self.assertIn('--min-tests "$MIN_TESTS"', report)
        self.assertNotIn("continue-on-error", inventory)
        # The count lists what the run runs.
        for listed, ran in (
            ("E2E_CONFIG: ${{ matrix.config }}", "--config ${{ matrix.config }}"),
            (
                "E2E_PLAYWRIGHT_ARGS: ${{ matrix.playwright_args }}",
                " ${{ matrix.playwright_args }} ",
            ),
            (
                "E2E_SHARD: ${{ matrix.shard_index }}/${{ matrix.shard_total }}",
                "format('--shard={0}/{1}', matrix.shard_index, matrix.shard_total)",
            ),
            (
                "E2E_REPEAT_EACH: ${{ matrix.suite == 'journeys' && "
                "inputs.journeys_repeat_each || 1 }}",
                "matrix.suite == 'journeys' && "
                "format('--repeat-each={0}', inputs.journeys_repeat_each || 1)",
            ),
        ):
            with self.subTest(listed=listed):
                self.assertIn(f"{listed}\n", inventory)
                self.assertIn(ran, run)
        self.assertIn("E2E_MIN_TESTS: ${{ matrix.min_tests }}\n", inventory)
        # Counted while the stack builds, so a short inventory fails early.
        steps = _e2e_step_names(workflow)
        self.assertLess(steps.index(INVENTORY_STEP), steps.index(STACK_WAIT_STEP))

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


# Stands in for `pnpm exec playwright test ... --list`: records its arguments
# and lists FAKE_TOTAL tests, or FAKE_SHARD for a --shard listing.
FAKE_PNPM = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS"
if [[ -n $FAKE_FAIL ]]; then
  echo "Error: broken-journey.spec.ts: Unexpected token"
  exit 1
fi
echo "Listing tests:"
if [[ -n $FAKE_NO_TOTAL ]]; then exit 0; fi
if [[ " $* " == *" --shard="* ]]; then n=$FAKE_SHARD; else n=$FAKE_TOTAL; fi
if (( n == 1 )); then echo "Total: 1 test in 1 file"; else echo "Total: $n tests in 3 files"; fi
"""


class E2EShardInventoryTests(unittest.TestCase):
    """Runs the "Count the shard's tests" step's script against a fake pnpm."""

    def _count(
        self,
        total: int,
        shard: int,
        min_tests: int | str,
        fail: bool = False,
        no_total: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], str, list[str]]:
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-fullstack-ci.yml"
        ).read_text(encoding="utf-8")
        script = textwrap.dedent(
            _step(workflow, INVENTORY_STEP).split("run: |\n", 1)[1]
        )
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            (folder / "pnpm").write_text(FAKE_PNPM, encoding="utf-8")
            (folder / "pnpm").chmod(0o755)
            output = folder / "github_output"
            output.touch()
            result = subprocess.run(
                # GitHub's `shell: bash`
                ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
                env={
                    **os.environ,
                    "PATH": f"{folder}{os.pathsep}{os.environ['PATH']}",
                    "GITHUB_OUTPUT": str(output),
                    "E2E_CONFIG": "playwright.journeys.config.ts",
                    "E2E_PLAYWRIGHT_ARGS": "--project hosted",
                    "E2E_MIN_TESTS": str(min_tests),
                    "E2E_SHARD": "2/2",
                    "E2E_REPEAT_EACH": "3",
                    "FAKE_CALLS": str(folder / "calls"),
                    "FAKE_TOTAL": str(total),
                    "FAKE_SHARD": str(shard),
                    "FAKE_FAIL": "1" if fail else "",
                    "FAKE_NO_TOTAL": "1" if no_total else "",
                },
                capture_output=True,
                text=True,
            )
            log = folder / "calls"
            calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
            return result, output.read_text(encoding="utf-8"), calls

    def test_outputs_the_shards_own_count(self):
        result, output, calls = self._count(total=41, shard=21, min_tests=41)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(output, "tests=21\n")
        listing = (
            "exec playwright test --config playwright.journeys.config.ts"
            " --project hosted --list --reporter=list"
        )
        # The whole suite as pinned, then this shard with the run's repeats.
        self.assertEqual(calls, [listing, f"{listing} --shard=2/2 --repeat-each=3"])

    def test_reads_a_single_test(self):
        result, output, _ = self._count(total=1, shard=1, min_tests=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(output, "tests=1\n")

    def test_fails_when_the_suite_lists_fewer_tests_than_the_pin(self):
        result, output, _ = self._count(total=40, shard=20, min_tests=41)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "::error title=Fewer tests than min_tests::The suite lists 40 tests,"
            " below its pinned min_tests of 41",
            result.stdout,
        )
        self.assertEqual(output, "")

    def test_fails_on_an_empty_shard(self):
        result, output, _ = self._count(total=41, shard=0, min_tests=41)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("::error title=Empty shard::", result.stdout)
        self.assertEqual(output, "")

    def test_fails_when_the_listing_has_no_total(self):
        result, output, _ = self._count(total=41, shard=21, min_tests=41, no_total=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("::error title=No test count::", result.stdout)
        self.assertEqual(output, "")

    def test_fails_when_min_tests_is_not_a_number(self):
        # An empty pin would make every comparison false and skip the floor.
        for pin in ("", "forty"):
            with self.subTest(pin=pin):
                result, output, calls = self._count(total=41, shard=21, min_tests=pin)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("::error title=No pinned min_tests::", result.stdout)
                self.assertEqual(output, "")
                self.assertEqual(calls, [])

    def test_fails_and_shows_why_when_listing_fails(self):
        result, output, _ = self._count(total=41, shard=21, min_tests=41, fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unexpected token", result.stderr)
        self.assertIn("::error title=Listing the tests failed::", result.stdout)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
