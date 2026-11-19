import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("ci_affected.py")
CONFIG = SCRIPT.parents[1] / "ci-projects.json"


class AffectedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.git("init", "-q")
        self.write("docs/start.md")
        self.base = self.commit()
        self.event = self.root / "event.json"
        self.plan = self.root / "plan.json"
        self.config = self.root / "projects.json"
        self.config.write_bytes(CONFIG.read_bytes())

    def git(self, *args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                f"core.hooksPath={self.root / 'no-hooks'}",
                "-c",
                "user.name=CI test",
                "-c",
                "user.email=ci@example.invalid",
                *args,
            ],
            cwd=self.repository,
            text=True,
            encoding="utf-8",
            stderr=subprocess.PIPE,
        ).strip()

    def write(self, path):
        target = self.repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "-qm", "ci: exercise affected-project selection")
        return self.git("rev-parse", "HEAD")

    def invoke(
        self, mode, *, event_name="pull_request", results=None, expected=0, base=None
    ):
        head = self.git("rev-parse", "HEAD")
        event = {
            "pull_request": {"base": {"sha": base or self.base}, "head": {"sha": head}},
            "merge_group": {"base_sha": base or self.base, "head_sha": head},
        }
        self.event.write_text(json.dumps(event))
        env = {
            key: value for key, value in os.environ.items() if key != "GITHUB_OUTPUT"
        }
        env.update(
            GITHUB_EVENT_NAME=event_name,
            GITHUB_EVENT_PATH=str(self.event),
            CI_RESULTS=json.dumps(results),
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                mode,
                "--plan",
                str(self.plan),
                "--config",
                str(self.config),
            ],
            cwd=self.repository,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_hidden_server_change_cannot_turn_all_skipped_jobs_green(self):
        self.write("autogpt_platform/backend/backend/server.py")
        self.commit()
        config = json.loads(self.config.read_text())
        config["backend"] = []
        config["fullstack"] = []
        config["unrelated"].append("autogpt_platform/backend/")
        self.config.write_text(json.dumps(config))
        self.invoke("plan")
        self.assertFalse(any(json.loads(self.plan.read_text()).values()))
        result = self.invoke(
            "gate",
            results={
                "changes": {"result": "success"},
                **{
                    name: {"result": "skipped"}
                    for name in ("backend", "frontend", "fullstack")
                },
            },
            expected=1,
        )
        self.assertIn("Affected projects incorrectly skipped", result.stderr)

    def test_selected_project_requires_success_not_missing_cancelled_or_skipped(self):
        self.write("autogpt_platform/frontend/src/page.tsx")
        self.commit()
        self.invoke("plan")
        self.invoke("verify")
        plan = json.loads(self.plan.read_text())
        self.assertEqual(plan, {"backend": False, "frontend": True, "fullstack": True})
        results = {
            name: {"result": "success"}
            for name in ("changes", "backend", "frontend", "fullstack")
        }
        results["backend"]["result"] = "skipped"
        self.invoke("gate", results=results)
        for state in ("skipped", "cancelled", "failure", None):
            with self.subTest(state=state):
                results["frontend"]["result"] = state
                self.invoke("gate", results=results, expected=1)

    def test_sparse_checkout_still_detects_and_requires_unmaterialized_changes(self):
        self.write(".github/selection.json")
        self.base = self.commit()
        self.write("autogpt_platform/backend/backend/server.py")
        self.commit()
        self.git("sparse-checkout", "set", "--no-cone", "/.github/")
        self.assertFalse((self.repository / "autogpt_platform").exists())
        for event in ("pull_request", "merge_group"):
            with self.subTest(event=event):
                self.invoke("plan", event_name=event)
                self.invoke("verify", event_name=event)
                plan = json.loads(self.plan.read_text())
                self.assertEqual(
                    plan, {"backend": True, "frontend": False, "fullstack": True}
                )
                self.plan.write_text(json.dumps({name: False for name in plan}))
                result = self.invoke(
                    "gate",
                    event_name=event,
                    results={
                        "changes": {"result": "success"},
                        **{name: {"result": "skipped"} for name in plan},
                    },
                    expected=1,
                )
                self.assertIn("Affected projects incorrectly skipped", result.stderr)

    def test_delete_and_rename_out_of_backend_keep_backend_required(self):
        for operation in ("delete", "rename"):
            with self.subTest(operation=operation):
                self.write("autogpt_platform/backend/old.py")
                base = self.commit()
                path = self.repository / "autogpt_platform/backend/old.py"
                if operation == "delete":
                    path.unlink()
                else:
                    path.rename(self.repository / "docs/renamed.md")
                self.commit()
                self.invoke("plan", base=base)
                self.invoke("verify", base=base)
                self.assertTrue(json.loads(self.plan.read_text())["backend"])

    def test_schema_shared_library_and_unknown_paths_select_dependents(self):
        for path, required in (
            (
                "autogpt_platform/backend/schema.prisma",
                {"backend", "frontend", "fullstack"},
            ),
            ("autogpt_platform/autogpt_libs/changed.py", {"backend", "fullstack"}),
            (
                "autogpt_platform/new-project/config.json",
                {"backend", "frontend", "fullstack"},
            ),
            (
                "autogpt_platform/docker-compose.yml",
                {"backend", "frontend", "fullstack"},
            ),
            (".github/workflows/platform-ci.yml", {"backend", "frontend", "fullstack"}),
        ):
            with self.subTest(path=path):
                base = self.git("rev-parse", "HEAD")
                self.write(path)
                self.commit()
                self.invoke("plan", base=base)
                self.invoke("verify", base=base)
                selected = {
                    name
                    for name, enabled in json.loads(self.plan.read_text()).items()
                    if enabled
                }
                self.assertEqual(selected, required)

    def test_docs_can_skip_but_selection_failure_still_fails_gate(self):
        self.write("docs/change.md")
        self.commit()
        self.invoke("plan")
        results = {
            name: {"result": "skipped"} for name in ("backend", "frontend", "fullstack")
        }
        results["changes"] = {"result": "success"}
        self.invoke("gate", results=results)
        results["changes"]["result"] = "failure"
        self.invoke("gate", results=results, expected=1)

    def test_merge_group_and_nightly_selection(self):
        self.write("autogpt_platform/backend/change.py")
        self.commit()
        self.invoke("plan", event_name="merge_group")
        self.invoke("verify", event_name="merge_group")
        self.assertTrue(json.loads(self.plan.read_text())["backend"])
        for event in ("schedule", "workflow_dispatch"):
            self.invoke("plan", event_name=event)
            self.invoke("verify", event_name=event)
            self.assertTrue(all(json.loads(self.plan.read_text()).values()))

    def test_base_only_changes_are_not_mistaken_for_pr_changes(self):
        self.write("autogpt_platform/backend/base-only.py")
        current_base = self.commit()
        self.git("checkout", "--detach", self.base)
        self.write("docs/pr.md")
        self.commit()
        self.invoke("plan", base=current_base)
        self.invoke("verify", base=current_base)
        self.assertFalse(any(json.loads(self.plan.read_text()).values()))

    def test_unavailable_base_and_malformed_selection_fail_closed(self):
        self.invoke("plan", base="f" * 40, expected=1)
        self.plan.write_text(
            '{"backend": "false", "frontend": false, "fullstack": false}'
        )
        self.invoke("verify", expected=1)

    def test_all_shards_skipped_fails_even_when_other_project_jobs_pass(self):
        results = {
            "lint": {"result": "success"},
            "integration_test": {"result": "skipped"},
        }
        for state in ("skipped", "failure", "cancelled", "success"):
            results["integration_test"]["result"] = state
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "jobs",
                    "--required-jobs",
                    "lint",
                    "integration_test",
                ],
                env={**os.environ, "CI_RESULTS": json.dumps(results)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0 if state == "success" else 1, result.stderr
            )


if __name__ == "__main__":
    unittest.main()
