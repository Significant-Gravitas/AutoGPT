import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import backend_test_shard
import yaml
from backend_test_shard import PARTITIONS, SHARDS, report_shards, shard_args


class BackendShardTests(unittest.TestCase):
    def test_all_test_locations_are_assigned_once_with_backend_config(self):
        paths = [
            "backend/data/sample_unit_test.py",
            "backend/copilot/sample_unit_test.py",
            "backend/copilot/dream/sample_unit_test.py",
            "backend/copilot/graphiti/sample_unit_test.py",
            "backend/copilot/graphiti/new_folder/sample_unit_test.py",
            "backend/copilot/billing_e2e/sample_unit_test.py",
            "backend/copilot/billing_e2e/thread_billing_test.py",
            "backend/copilot/billing_e2e/new_folder/sample_unit_test.py",
            "backend/copilot/sdk/sample_unit_test.py",
            "backend/copilot/sdk/new_folder/sample_unit_test.py",
            "backend/copilot/sdk_sibling/sample_unit_test.py",
            "backend/copilot/graphiti_sibling/sample_unit_test.py",
            "backend/copilot/tools/sample_unit_test.py",
            "backend/copilot/sdk_like/sample_unit_test.py",
            "backend/util/sample_unit_test.py",
            "backend/executor/sample_unit_test.py",
            "backend/executor/billing_e2e/sample_unit_test.py",
            "backend/api/sample_unit_test.py",
            "backend/api/features/experts/sample_unit_test.py",
            "backend/api/features/experts/new_folder/sample_unit_test.py",
            "backend/api/features/experts_sibling/sample_unit_test.py",
            "backend/blocks/sample_unit_test.py",
            "backend/new_folder/sample_unit_test.py",
            "backend/new_unit_test.py",
            "new_top_level/sample_unit_test.py",
            "root_unit_test.py",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "autogpt_platform/backend"
            backend.mkdir(parents=True)
            (backend / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\npython_files = ["*_test.py"]\n'
            )
            helpers = root / ".github/scripts"
            helpers.mkdir(parents=True)
            (helpers / "helper_unit_test.py").write_text("def test_helper(): pass\n")
            for index, path in enumerate(paths):
                target = backend / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"def test_case_{index}(): pass\n")
            names = []
            assignments = {}
            for shard in (*SHARDS, "remainder"):
                report = root / f"{shard}.xml"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "--import-mode=importlib",
                        *shard_args(shard),
                        f"--junitxml={report}",
                        "-q",
                    ],
                    cwd=backend,
                    env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for case in ElementTree.parse(report).iter("testcase"):
                    name = case.get("name")
                    names.append(name)
                    assignments[name] = shard
            self.assertCountEqual(
                names, [f"test_case_{i}" for i in range(len(paths))] + ["test_helper"]
            )
            separate_jobs = [
                "backend/data/sample_unit_test.py",
                "backend/copilot/sample_unit_test.py",
                "backend/copilot/dream/sample_unit_test.py",
                "backend/copilot/sdk/sample_unit_test.py",
                "backend/copilot/billing_e2e/thread_billing_test.py",
                "backend/util/sample_unit_test.py",
                "backend/api/sample_unit_test.py",
                "backend/blocks/sample_unit_test.py",
            ]
            self.assertEqual(
                len(
                    {
                        assignments[f"test_case_{paths.index(path)}"]
                        for path in separate_jobs
                    }
                ),
                len(separate_jobs),
            )

    def test_a_shard_leaves_out_the_folders_a_nested_shard_runs(self):
        self.assertEqual(
            shard_args("copilot"),
            [
                "-c",
                "pyproject.toml",
                "backend/copilot",
                "--ignore=backend/copilot/dream",
                "--ignore=backend/copilot/graphiti",
                "--ignore=backend/copilot/sdk",
                "--ignore=backend/copilot/billing_e2e",
                "--ignore=backend/copilot/billing_e2e/thread_billing_test.py",
            ],
        )
        self.assertEqual(
            shard_args("sdk-billing"),
            [
                "-c",
                "pyproject.toml",
                "backend/copilot/sdk",
                "backend/copilot/billing_e2e",
                "--ignore=backend/copilot/billing_e2e/thread_billing_test.py",
            ],
        )

    def test_unknown_shard_is_rejected(self):
        with self.assertRaises(ValueError):
            shard_args("typo")

    def test_workflow_matrix_matches_script_shards(self):
        workflow = (
            Path(__file__).resolve().parents[1] / "workflows/platform-backend-ci.yml"
        )
        matrix = yaml.safe_load(workflow.read_text())["jobs"]["test"]["strategy"][
            "matrix"
        ]["test-shard"]
        self.assertCountEqual(matrix, [*SHARDS, "remainder"])

    def test_main_propagates_pytest_status_and_scopes_docker_tests_to_their_shard(
        self,
    ):
        flags = ("E2E_REDIS_CLUSTER_RESTART", "E2E_CLAMD_CONTAINER")
        for shard, enabled in (
            ("data", {"E2E_REDIS_CLUSTER_RESTART"}),
            ("util-executor", {"E2E_CLAMD_CONTAINER"}),
            ("copilot", set()),
            ("memory-experts", set()),
            ("sdk-billing", set()),
            ("billing-threads", set()),
            ("api", set()),
            ("remainder", set()),
        ):
            with (
                self.subTest(shard=shard),
                patch.object(
                    backend_test_shard.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 3),
                ) as run,
                patch.dict(os.environ),
                patch.object(sys, "argv", ["backend_test_shard.py", shard, "--", "-q"]),
            ):
                for flag in (*flags, "PYRO_CLIENT_COMM_RETRY"):
                    os.environ.pop(flag, None)
                self.assertEqual(backend_test_shard.main(), 3)
                self.assertEqual(
                    {f for f in flags if os.environ.get(f) == "1"}, enabled
                )
                self.assertEqual(
                    os.environ.get("PYRO_CLIENT_COMM_RETRY"),
                    "3" if shard == "sdk-billing" else None,
                )
                run.assert_called_once_with(
                    [
                        "poetry",
                        "run",
                        "pytest",
                        "-p",
                        "backend_shard_plugin",
                        *shard_args(shard),
                        "-q",
                    ],
                    check=False,
                )

    def test_weighted_layout_keeps_both_docker_suites_enabled(self):
        with (
            patch.object(
                backend_test_shard.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            patch.dict(os.environ),
            patch.object(
                sys,
                "argv",
                ["backend_test_shard.py", "--layout", "weighted", "sdk-billing"],
            ),
        ):
            for flag in (
                "E2E_REDIS_CLUSTER_RESTART",
                "E2E_CLAMD_CONTAINER",
                "PYRO_CLIENT_COMM_RETRY",
            ):
                os.environ.pop(flag, None)
            self.assertEqual(backend_test_shard.main(), 0)
            self.assertEqual(os.environ["E2E_REDIS_CLUSTER_RESTART"], "1")
            self.assertEqual(os.environ["E2E_CLAMD_CONTAINER"], "1")
            self.assertNotIn("PYRO_CLIENT_COMM_RETRY", os.environ)

    def test_every_shard_configures_isolated_services_and_unknown_shards_fail(self):
        workflow = yaml.safe_load(
            (
                Path(__file__).resolve().parents[1]
                / "workflows/platform-backend-ci.yml"
            ).read_text()
        )
        configure = next(
            step
            for step in workflow["jobs"]["test"]["steps"]
            if step.get("name") == "Configure test shard"
        )["run"]
        databases = set()
        vhosts = set()
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / "environment"
            for shard in (*SHARDS, "remainder", "unknown"):
                with self.subTest(shard=shard):
                    environment.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", "-e", "-o", "pipefail", "-c", configure],
                        env={
                            **os.environ,
                            "MATRIX_TEST_SHARD": shard,
                            "MATRIX_TEST_PARTITION": "1",
                            "GITHUB_ENV": str(environment),
                        },
                        capture_output=True,
                        text=True,
                    )
                    if shard == "unknown":
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(environment.exists())
                        continue
                    self.assertEqual(result.returncode, 0, result.stderr)
                    values = dict(
                        line.split("=", 1)
                        for line in environment.read_text().splitlines()
                    )
                    self.assertEqual(values["TEST_SHARD"], f"{shard}-1")
                    self.assertRegex(values["DATABASE_NAME"], r"^[a-z_][a-z0-9_]*$")
                    self.assertEqual(
                        values["DATABASE_NAME"] == "postgres", shard == "data"
                    )
                    self.assertNotIn(values["DATABASE_NAME"], databases)
                    self.assertNotIn(values["RABBITMQ_VHOST"], vhosts)
                    databases.add(values["DATABASE_NAME"])
                    vhosts.add(values["RABBITMQ_VHOST"])

    def test_coverage_waits_for_every_matrix_upload(self):
        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load(
            (root / ".github/workflows/platform-backend-ci.yml").read_text()
        )
        matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
        codecov = yaml.safe_load((root / "codecov.yml").read_text())
        jobs = [
            {"python-version": version, "test-shard": shard, "partition": part}
            for version in matrix["python-version"]
            for shard in matrix["test-shard"]
            for part in matrix["partition"]
        ]
        jobs = [
            job
            for job in jobs
            if not any(
                all(job[key] == value for key, value in excluded.items())
                for excluded in matrix.get("exclude", [])
            )
        ]
        jobs.extend(matrix.get("include", []))
        identities = [
            (job["python-version"], f"{job['test-shard']}-{job['partition']}")
            for job in jobs
        ]
        expected = [
            (version, shard)
            for version in matrix["python-version"]
            for shard in report_shards(partitioned=True)
        ]
        self.assertCountEqual(identities, expected)
        self.assertEqual(
            codecov["flags"]["platform-backend"]["after_n_builds"],
            len(expected),
        )
