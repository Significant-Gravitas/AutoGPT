import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1]
COPY_MATRIX = (
    "${{ fromJSON(github.event_name == 'pull_request' && '[1,2,3]' || '[1]') }}"
)
TEST_JOBS = {
    "platform-backend-ci.yml": {"autogpt-libs-test": 1, "test": 12},
    "platform-frontend-ci.yml": {
        "tooltip-browser": 1,
        "lint": 1,
        "integration_test": 4,
    },
    "platform-fullstack-ci.yml": {"check-api-types": 1, "e2e_test": 2},
    "platform-installer-ci.yml": {"shell": 1, "macos": 1, "windows": 1},
    "platform-single-container-docker.yml": {"build-and-scan": 2},
    "platform-swap-proxy-ci.yml": {"test": 1},
    "classic-autogpt-ci.yml": {"test": 1},
    "classic-autogpt-docker-ci.yml": {"test": 1},
    "classic-forge-ci.yml": {"test": 1},
    "classic-autogpts-ci.yml": {"serve-agent-protocol": 1},
    "classic-benchmark-ci.yml": {"benchmark-tests": 1},
    "repo-workflow-checker.yml": {"test-ci": 1},
}


def load_workflows():
    return {
        name: yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
        for name in TEST_JOBS
    }


def expand_matrix(matrix, event):
    axes = {key: value for key, value in matrix.items() if key != "include"}
    if axes["test-copy"] != COPY_MATRIX:
        raise ValueError("test-copy must select three copies only for pull_request")
    axes["test-copy"] = [1, 2, 3] if event == "pull_request" else [1]
    combinations = []
    for values in itertools.product(*axes.values()):
        original = dict(zip(axes, values))
        combination = original.copy()
        for extra in matrix.get("include", []):
            if all(
                key not in original or original[key] == value
                for key, value in extra.items()
            ):
                combination.update(extra)
        combinations.append(combination)
    return combinations


def render_name(template, matrix, event):
    label = re.compile(
        r"\$\{\{ github.event_name == 'pull_request' && format\('([^']+)', matrix.test-copy\) \|\| '' \}\}"
    )
    rendered = label.sub(
        lambda match: (
            match[1].format(matrix["test-copy"]) if event == "pull_request" else ""
        ),
        template,
    )
    for key, value in matrix.items():
        rendered = rendered.replace("${{ matrix.%s }}" % key, str(value))
    if "${{" in rendered:
        raise ValueError(f"Unresolved name: {rendered}")
    return rendered


class TestCopyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflows = load_workflows()

    def test_every_existing_leg_gets_three_pr_copies_and_one_for_other_events(self):
        for event in (
            "pull_request",
            "push",
            "merge_group",
            "workflow_dispatch",
            "schedule",
            "release",
        ):
            expected_copies = [1, 2, 3] if event == "pull_request" else [1]
            for filename, jobs in TEST_JOBS.items():
                for job_id, leg_count in jobs.items():
                    with self.subTest(event=event, workflow=filename, job=job_id):
                        job = self.workflows[filename]["jobs"][job_id]
                        self.assertFalse(job["strategy"]["fail-fast"])
                        self.assertNotIn("max-parallel", job["strategy"])
                        self.assertNotIn("concurrency", job)
                        self.assertFalse(job.get("continue-on-error", False))
                        combinations = expand_matrix(job["strategy"]["matrix"], event)
                        self.assertEqual(
                            len(combinations), leg_count * len(expected_copies)
                        )
                        for copy in expected_copies:
                            self.assertEqual(
                                sum(row["test-copy"] == copy for row in combinations),
                                leg_count,
                            )

    def test_each_copy_has_a_distinct_check_and_artifact_name(self):
        check_names = set()
        for filename, jobs in TEST_JOBS.items():
            artifacts = set()
            for job_id in jobs:
                job = self.workflows[filename]["jobs"][job_id]
                for row in expand_matrix(job["strategy"]["matrix"], "pull_request"):
                    name = render_name(job["name"], row, "pull_request")
                    self.assertNotIn(name, check_names)
                    self.assertIn(f"copy {row['test-copy']}/3", name)
                    check_names.add(name)
                    for step in job["steps"]:
                        if not step.get("uses", "").startswith(
                            "actions/upload-artifact@"
                        ):
                            continue
                        artifact = render_name(
                            step["with"]["name"], row, "pull_request"
                        )
                        self.assertNotIn(artifact, artifacts)
                        self.assertIn(f"copy-{row['test-copy']}", artifact)
                        artifacts.add(artifact)

    def test_include_metadata_survives_crossing_suites_and_architectures_with_copies(
        self,
    ):
        e2e = self.workflows["platform-fullstack-ci.yml"]["jobs"]["e2e_test"]
        appliance = self.workflows["platform-single-container-docker.yml"]["jobs"][
            "build-and-scan"
        ]
        for job, axis in ((e2e, "suite"), (appliance, "platform")):
            matrix = job["strategy"]["matrix"]
            expected = {row[axis]: row for row in matrix["include"]}
            self.assertCountEqual(matrix[axis], expected)
            for row in expand_matrix(matrix, "pull_request"):
                self.assertEqual(
                    {key: value for key, value in row.items() if key != "test-copy"},
                    expected[row[axis]],
                )

    def test_release_publication_and_status_gate_are_not_repeated(self):
        appliance = self.workflows["platform-single-container-docker.yml"]["jobs"]
        for job_id in (
            "authorize-publication",
            "publish-platform-digests",
            "publish-manifest",
        ):
            self.assertNotIn(
                "test-copy", appliance[job_id].get("strategy", {}).get("matrix", {})
            )
        checker = self.workflows["repo-workflow-checker.yml"]["jobs"]["status-check"]
        self.assertNotIn("strategy", checker)
        self.assertEqual(checker["name"], "Check PR Status")
        self.assertEqual(
            checker["steps"][0]["with"]["ref"],
            "${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha }}",
        )


class TriplicateGateEndToEndTests(unittest.TestCase):
    def run_gate(self, checks, event_name="pull_request"):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if "/check-runs" in self.path:
                    if "page=2" in self.path:
                        payload = {"check_runs": checks[100:]}
                    else:
                        payload = {"check_runs": checks[:100]}
                else:
                    payload = {"workflow_runs": []}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if (
                    "/check-runs" in self.path
                    and len(checks) > 100
                    and "page=2" not in self.path
                ):
                    self.send_header(
                        "Link",
                        f'<http://127.0.0.1:{self.server.server_port}/check-runs?page=2>; rel="next"',
                    )
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    event_path = Path(directory) / "event.json"
                    event = (
                        {"pull_request": {"head": {"sha": "abc123"}}}
                        if event_name == "pull_request"
                        else {"merge_group": {}}
                    )
                    event_path.write_text(json.dumps(event), encoding="utf-8")
                    return subprocess.run(
                        [
                            sys.executable,
                            str(WORKFLOWS / "scripts/check_actions_status.py"),
                        ],
                        env={
                            **os.environ,
                            "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
                            "GITHUB_EVENT_PATH": str(event_path),
                            "GITHUB_EVENT_NAME": event_name,
                            "GITHUB_REPOSITORY": "example/repository",
                            "GITHUB_SHA": "abc123",
                            "GITHUB_TOKEN": "",
                            "GITHUB_RUN_ID": "1",
                            "NO_PROXY": "127.0.0.1",
                        },
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_each_of_three_copies_must_pass_even_across_api_pages(self):
        workflows = load_workflows()
        checks = []
        for filename, jobs in TEST_JOBS.items():
            for job_id in jobs:
                job = workflows[filename]["jobs"][job_id]
                for row in expand_matrix(job["strategy"]["matrix"], "pull_request"):
                    checks.append(
                        {
                            "id": len(checks),
                            "name": render_name(job["name"], row, "pull_request"),
                            "status": "completed",
                            "conclusion": "success",
                        }
                    )
        self.assertGreater(len(checks), 100)
        passing = self.run_gate(checks)
        self.assertEqual(passing.returncode, 0, passing.stdout + passing.stderr)
        for copy in (1, 2, 3):
            name = f"end-to-end tests (journeys) (copy {copy}/3)"
            failed = next(check for check in checks if check["name"] == name)
            for conclusion in ("failure", "cancelled", "timed_out"):
                with self.subTest(copy=copy, conclusion=conclusion):
                    mixed = [check for check in checks if check is not failed] + [
                        {**failed, "conclusion": conclusion}
                    ]
                    result = self.run_gate(mixed)
                    self.assertEqual(
                        result.returncode, 1, result.stdout + result.stderr
                    )
                    self.assertIn(name, result.stdout)

    def test_single_merge_queue_journeys_check_keeps_its_existing_advisory_policy(self):
        result = self.run_gate(
            [
                {
                    "id": 1,
                    "name": "end-to-end tests (journeys)",
                    "status": "completed",
                    "conclusion": "failure",
                }
            ],
            "merge_group",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
