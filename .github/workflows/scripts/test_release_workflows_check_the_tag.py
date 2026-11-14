"""Every workflow a published release starts must check which release it is.

This repository publishes releases for more than the platform: the rolling
preview-seed-fixture, and Local PC companion releases to come. A workflow on
``release: published`` that never looks at the tag runs for all of them. The
production deploy did, so any published release would have deployed prod.
"""

import os
import pathlib
import subprocess
import unittest

import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1]
TAG_CHECKS = ("github.event.release.tag_name", "refs/tags/")
UNPREFIXED_GUARD = "reject-unprefixed-platform-tag"


def triggers(workflow: dict) -> set[str]:
    """The events a workflow runs on, in any of the forms YAML allows."""
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, (list, dict)):
        return set(on)
    return set()


def reads_the_tag(workflow: dict) -> bool:
    """Whether a job's or a step's condition reads the release tag, directly
    or through a job output that a step computed from it (the single-container
    workflow's publish policy). A mention anywhere else, such as a build
    report's source URL, decides nothing."""
    jobs = workflow.get("jobs") or {}
    decided_by_tag = set()
    for name, job in jobs.items():
        tag_steps = {
            step["id"]
            for step in job.get("steps") or []
            if step.get("id")
            and any(
                check in str(value)
                for value in (step.get("env") or {}).values()
                for check in TAG_CHECKS
            )
        }
        for output, expression in (job.get("outputs") or {}).items():
            if any(f"steps.{step}.outputs." in str(expression) for step in tag_steps):
                decided_by_tag.add(f"needs.{name}.outputs.{output}")
    for job in jobs.values():
        conditions = [
            job.get("if"),
            *(step.get("if") for step in job.get("steps") or []),
        ]
        for condition in filter(None, conditions):
            if any(check in str(condition) for check in TAG_CHECKS) or any(
                output in str(condition) for output in decided_by_tag
            ):
                return True
    return False


class ReleaseWorkflowsCheckTheTag(unittest.TestCase):
    def test_every_release_workflow_checks_the_tag(self) -> None:
        unchecked = [
            path.name
            for path in sorted(WORKFLOWS.glob("*.y*ml"))
            if "release" in triggers(workflow := yaml.safe_load(path.read_text()))
            and not reads_the_tag(workflow)
        ]
        self.assertEqual(
            unchecked,
            [],
            "These workflows run for every published release. Check the tag, "
            "e.g. startsWith(github.event.release.tag_name, 'autogpt-platform-').",
        )

    def test_only_a_deciding_mention_of_the_tag_counts(self) -> None:
        # Only a condition (`if:`) that reads the tag gates a release. A
        # mention anywhere else, such as a build report's source_url or an
        # image label, decides nothing, so it must not count.
        for ungated in (
            "on: release\njobs:\n  build:\n    runs-on: x\n    steps:\n"
            "      - name: Generate build report\n        run: ./report.sh\n        env:\n"
            "          source_url: ${{ github.event.release.tag_name }}\n",
            "on: release\njobs:\n  build:\n    runs-on: x\n    steps:\n"
            "      - uses: docker/metadata-action@v5\n        with:\n"
            "          labels: source=${{ github.event.release.tag_name }}\n",
        ):
            self.assertFalse(reads_the_tag(yaml.safe_load(ungated)), ungated)
        for gated in (
            "on: release\njobs:\n  build:\n"
            "    if: startsWith(github.ref, 'refs/tags/autogpt-v')\n    runs-on: x\n",
            "on: release\njobs:\n  build:\n    runs-on: x\n    steps:\n"
            "      - if: startsWith(github.event.release.tag_name, 'x')\n        run: true\n",
            # A policy step reads the tag, and its output gates the publish.
            "on: release\njobs:\n  authorize:\n    runs-on: x\n"
            "    outputs:\n      allowed: ${{ steps.policy.outputs.allowed }}\n"
            "    steps:\n      - id: policy\n        run: ./decide.sh\n        env:\n"
            "          TAG: ${{ github.event.release.tag_name }}\n"
            "  publish:\n    needs: authorize\n    runs-on: x\n"
            "    if: needs.authorize.outputs.allowed == 'true'\n",
        ):
            self.assertTrue(reads_the_tag(yaml.safe_load(gated)), gated)

    def test_every_form_of_the_release_trigger_counts(self) -> None:
        for source in (
            "on: release",
            "on: [push, release]",
            "on: {release: {types: [published]}}",
            "on:\n  release:  # a comment\n    types: [published]",
            "on:\n    release:\n      types: [published]",
        ):
            self.assertIn("release", triggers(yaml.safe_load(source)), source)
        job_named_release = "on: push\njobs:\n  release:\n    runs-on: ubuntu-latest"
        self.assertNotIn("release", triggers(yaml.safe_load(job_named_release)))

    def test_production_deploys_only_platform_releases(self) -> None:
        workflow = yaml.safe_load(
            (WORKFLOWS / "platform-autogpt-deploy-prod.yml").read_text()
        )
        jobs = workflow["jobs"]
        # Exactly this: an added `|| true` would keep both clauses and still
        # deploy every published release.
        self.assertEqual(
            jobs["migrate"]["if"],
            "github.event_name == 'workflow_dispatch' || "
            "startsWith(github.event.release.tag_name, 'autogpt-platform-')",
        )
        # Manual dispatch is the only way to deploy outside a platform release.
        self.assertIn("workflow_dispatch", triggers(workflow))
        # The jobs after it are skipped with it only while they need it and
        # have no condition of their own: `if: always()` on `trigger` would
        # deploy for every published release. Every job, so one added later
        # without `needs: migrate` can't run on every release unnoticed.
        for job in sorted(set(jobs) - {"migrate", UNPREFIXED_GUARD}):
            needs = jobs[job].get("needs", [])
            # A list, so a string `needs` can't pass on a substring (`pre-migrate`).
            self.assertIn("migrate", [needs] if isinstance(needs, str) else needs, job)
            self.assertNotIn("if", jobs[job], job)

    def test_a_platform_release_tagged_without_the_prefix_fails(self) -> None:
        guard = yaml.safe_load(
            (WORKFLOWS / "platform-autogpt-deploy-prod.yml").read_text()
        )["jobs"][UNPREFIXED_GUARD]
        self.assertEqual(
            guard["if"],
            "github.event_name == 'release' && "
            "startsWith(github.event.release.tag_name, 'v')",
        )
        # It runs where migrate doesn't, so it must not be able to deploy.
        self.assertNotIn("needs", guard)
        self.assertNotIn("environment", guard)
        self.assertEqual(guard["permissions"], {})
        self.assertNotIn("secrets.", yaml.safe_dump(guard))
        [step] = guard["steps"]
        for tag, fails in (
            ("v0.6.44", True),
            ("v1.0.0", True),
            ("vision-preview", False),
        ):
            result = subprocess.run(
                ["bash", "-c", step["run"]],
                env={**os.environ, "TAG": tag},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode != 0, fails, tag)


if __name__ == "__main__":
    unittest.main()
