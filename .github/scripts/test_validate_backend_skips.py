import fnmatch
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import yaml

from backend_test_shard import SHARDS
from validate_backend_skips import main
from validate_junit import load_skip_policy
from validate_junit import main as validate_junit_main


class AggregateSkipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.reports = self.directory / "reports"
        self.policy_file = self.directory / "policy.json"
        self.policy = {
            "common": ["backend.example.test_shared"],
            "python_versions": {"3.11": [], "3.12": []},
        }
        for version in self.policy["python_versions"]:
            for shard in (*SHARDS, "remainder"):
                skips = ["test_shared"] if shard == "remainder" else []
                self.write_report(version, shard, skips)

    def write_report(self, version, shard, skips):
        report = self.report_path(version, shard)
        report.parent.mkdir(parents=True, exist_ok=True)
        root = ElementTree.Element("testsuite", name=shard)
        ElementTree.SubElement(
            root, "testcase", classname="backend.example", name=f"{shard}_passes"
        )
        for name in skips:
            case = ElementTree.SubElement(
                root, "testcase", classname="backend.example", name=name
            )
            ElementTree.SubElement(case, "skipped")
        ElementTree.ElementTree(root).write(report)
        return report

    def report_path(self, version, shard):
        return (
            self.reports
            / f"backend-test-reports-py{version}-{shard}"
            / f"junit-{shard}.xml"
        )

    def run_validator(self):
        self.policy_file.write_text(json.dumps(self.policy))
        with patch("sys.stderr") as stderr:
            result = main(
                [
                    "--allow-skips-from",
                    str(self.policy_file),
                    "--reports-dir",
                    str(self.reports),
                ]
            )
        return result, "".join(call.args[0] for call in stderr.write.call_args_list)

    def test_entry_used_only_by_a_different_shard_passes(self):
        self.assertEqual(self.run_validator()[0], 0)

    def test_entry_used_by_no_shard_fails_with_actionable_diagnostic(self):
        self.policy["common"].append("backend.example.test_removed")
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("Python 3.11", diagnostic)
        self.assertIn("stale", diagnostic)
        self.assertIn("backend.example.test_removed", diagnostic)
        self.assertIn("policy.json", diagnostic)

    def test_a_common_entry_cannot_be_hidden_by_another_python_version(self):
        self.write_report("3.12", "remainder", [])
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("Python 3.12", diagnostic)
        self.assertIn("backend.example.test_shared", diagnostic)

    def test_explicit_python_version_specific_skip_passes(self):
        self.policy["python_versions"]["3.12"] = ["backend.example.test_version_only"]
        self.write_report("3.12", "data", ["test_version_only"])
        self.assertEqual(self.run_validator()[0], 0)

    def test_same_skip_can_be_explicit_for_multiple_versions(self):
        for version in self.policy["python_versions"]:
            self.policy["python_versions"][version] = [
                "backend.example.test_version_only"
            ]
            self.write_report(version, "data", ["test_version_only"])
        self.assertEqual(self.run_validator()[0], 0)

    def test_python_specific_allowance_cannot_permit_skip_in_another_version(self):
        self.policy["python_versions"]["3.12"] = ["backend.example.test_version_only"]
        self.write_report("3.11", "data", ["test_version_only"])
        self.write_report("3.12", "data", ["test_version_only"])
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("unapproved skipped test IDs", diagnostic)
        self.assertIn("Python 3.11", diagnostic)

    def test_missing_shard_report_fails_even_when_all_skips_are_present(self):
        self.report_path("3.12", "copilot").unlink()
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("Python 3.12, shard copilot", diagnostic)
        self.assertIn("missing or empty", diagnostic)

    def test_missing_entire_python_version_fails(self):
        for shard in (*SHARDS, "remainder"):
            report = self.report_path("3.12", shard)
            report.unlink()
            report.parent.rmdir()
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("missing", diagnostic)
        self.assertIn("py3.12", diagnostic)

    def test_unknown_or_duplicate_shard_artifact_fails(self):
        self.write_report("3.11", "data-copy", [])
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("unexpected", diagnostic)
        self.assertIn("data-copy", diagnostic)

    def test_unconfigured_python_version_reports_fail(self):
        self.write_report("3.13", "data", [])
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("unexpected", diagnostic)
        self.assertIn("py3.13", diagnostic)

    def test_invalid_or_incomplete_reports_fail(self):
        cases = (
            ("", "missing or empty"),
            ("<testsuite>", "no element found"),
            ('<testsuite tests="0"/>', "no test cases"),
            (
                '<testsuite tests="2"><testcase name="only"/></testsuite>',
                "count is 2",
            ),
            (
                '<testsuite><testcase name="bad"><failure/></testcase></testsuite>',
                "1 failures",
            ),
        )
        for content, expected in cases:
            with self.subTest(content=content):
                self.report_path("3.11", "data").write_text(content)
                result, diagnostic = self.run_validator()
                self.assertEqual(result, 1)
                self.assertIn(expected, diagnostic)

    def test_new_skip_remains_rejected(self):
        self.write_report("3.11", "data", ["test_unapproved"])
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("unapproved skipped test IDs", diagnostic)

    def test_secret_gated_skip_passes_whether_or_not_the_run_had_the_secret(self):
        self.policy["secret_gated"] = {
            "EXAMPLE_API_KEY": ["backend.example.test_needs_secret"]
        }
        self.write_report("3.11", "data", ["test_needs_secret"])
        self.assertEqual(self.run_validator()[0], 0)
        self.write_report("3.11", "data", [])
        self.assertEqual(self.run_validator()[0], 0)

    def test_aggregate_validation_requires_explicit_versions(self):
        self.policy = ["backend.example.test_shared"]
        result, diagnostic = self.run_validator()
        self.assertEqual(result, 1)
        self.assertIn("explicit python_versions", diagnostic)

    def test_per_shard_validator_uses_the_same_version_specific_policy(self):
        self.policy["python_versions"]["3.12"] = ["backend.example.test_version_only"]
        report = self.write_report("3.12", "data", ["test_version_only"])
        self.policy_file.write_text(json.dumps(self.policy))
        arguments = ["--allow-skips-from", str(self.policy_file), str(report)]
        self.assertEqual(
            validate_junit_main(arguments + ["--python-version", "3.12"]), 0
        )
        with patch("sys.stderr"):
            self.assertEqual(
                validate_junit_main(arguments + ["--python-version", "3.11"]), 1
            )
            self.assertEqual(validate_junit_main(arguments), 1)
            self.assertEqual(
                validate_junit_main(arguments + ["--python-version", "3.13"]), 1
            )

    def test_invalid_policy_is_rejected_before_validation(self):
        invalid = (
            {},
            {"common": [], "python_versions": {}},
            {"common": [], "python_versions": {"3.11.1": []}},
            {
                "common": ["backend.test", "backend.test"],
                "python_versions": {"3.11": []},
            },
            {
                "common": [],
                "python_versions": {"3.11": ["backend.test", "backend.test"]},
            },
            {"common": ["backend.test"], "python_versions": {"3.11": ["backend.test"]}},
            {"common": [], "python_versions": {"3.11": [""]}},
            {"common": [], "python_versions": {"3.11": []}, "typo": []},
            {"common": [], "python_versions": {"3.11": []}, "secret_gated": []},
            {
                "common": [],
                "python_versions": {"3.11": []},
                "secret_gated": {"lowercase_key": ["backend.test"]},
            },
            {
                "common": ["backend.test"],
                "python_versions": {"3.11": []},
                "secret_gated": {"EXAMPLE_API_KEY": ["backend.test"]},
            },
            {
                "common": [],
                "python_versions": {"3.11": ["backend.test"]},
                "secret_gated": {"EXAMPLE_API_KEY": ["backend.test"]},
            },
            {
                "common": [],
                "python_versions": {"3.11": []},
                "secret_gated": {"EXAMPLE_API_KEY": ["backend.test", "backend.test"]},
            },
        )
        for policy in invalid:
            with self.subTest(policy=policy):
                self.policy = policy
                self.assertEqual(self.run_validator()[0], 1)


class WorkflowSkipPolicyTests(unittest.TestCase):
    def setUp(self):
        self.github = Path(__file__).resolve().parents[1]
        self.workflow = yaml.safe_load(
            (self.github / "workflows/platform-backend-ci.yml").read_text()
        )

    def test_explicit_policy_versions_match_every_backend_python_matrix(self):
        policy = load_skip_policy(self.github / "scripts/backend-allowed-skips.json")
        for name in ("test", "type-check"):
            versions = self.workflow["jobs"][name]["strategy"]["matrix"][
                "python-version"
            ]
            self.assertCountEqual(versions, policy.python_versions)

    def test_aggregate_job_uses_unmerged_artifacts_from_every_matrix_shard(self):
        jobs = self.workflow["jobs"]
        aggregate = jobs["validate-skip-policy"]
        self.assertEqual(aggregate["needs"], "test")
        self.assertEqual(aggregate["if"], "${{ always() }}")
        self.assertEqual(aggregate["runs-on"], "ubuntu-latest")
        download = next(
            step
            for step in aggregate["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        )["with"]
        self.assertFalse(download.get("merge-multiple", False))
        upload = next(
            step
            for step in jobs["test"]["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        )["with"]["name"]
        matrix = jobs["test"]["strategy"]["matrix"]
        for version in matrix["python-version"]:
            for shard in matrix["test-shard"]:
                artifact = upload.replace(
                    "${{ matrix.python-version }}", version
                ).replace("${{ matrix.test-shard }}", shard)
                self.assertEqual(artifact, f"backend-test-reports-py{version}-{shard}")
                self.assertTrue(fnmatch.fnmatchcase(artifact, download["pattern"]))

    def test_aggregate_step_refuses_partial_reports_from_unsuccessful_jobs(self):
        job = self.workflow["jobs"]["validate-skip-policy"]
        validate = job["steps"][-1]
        self.assertEqual(validate["if"], "${{ always() }}")
        self.assertEqual(
            validate["env"]["SHARD_JOB_RESULT"], "${{ needs.test.result }}"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python3"
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > invocation\n')
            python.chmod(0o755)
            for result in ("failure", "cancelled", "skipped", "success"):
                with self.subTest(result=result):
                    invocation = root / "invocation"
                    invocation.unlink(missing_ok=True)
                    run = subprocess.run(
                        ["bash", "-e", "-o", "pipefail", "-c", validate["run"]],
                        cwd=root,
                        env={
                            **os.environ,
                            "PATH": f"{root}:{os.environ['PATH']}",
                            "SHARD_JOB_RESULT": result,
                        },
                        capture_output=True,
                        text=True,
                    )
                    if result == "success":
                        self.assertEqual(run.returncode, 0, run.stderr)
                        arguments = invocation.read_text().splitlines()
                        self.assertEqual(
                            arguments[0], ".github/scripts/validate_backend_skips.py"
                        )
                        self.assertEqual(arguments[-1], "backend-shard-reports")
                    else:
                        self.assertNotEqual(run.returncode, 0)
                        self.assertIn("reports may be incomplete", run.stderr)
                        self.assertFalse(invocation.exists())

    def test_every_secret_gated_variable_reaches_the_test_job_as_a_secret(self):
        policy = load_skip_policy(self.github / "scripts/backend-allowed-skips.json")
        self.assertTrue(policy.secret_gated)
        environment = self.workflow["jobs"]["test"]["env"]
        for variable in policy.secret_gated:
            self.assertEqual(environment.get(variable), "${{ secrets.%s }}" % variable)

    def test_new_helper_and_test_changes_trigger_backend_ci(self):
        events = self.workflow["on"] if "on" in self.workflow else self.workflow[True]
        for event in ("push", "pull_request"):
            for name in ("validate_backend_skips.py", "test_validate_backend_skips.py"):
                self.assertIn(f".github/scripts/{name}", events[event]["paths"])
