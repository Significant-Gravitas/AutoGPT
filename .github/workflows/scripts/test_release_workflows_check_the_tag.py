"""Every workflow a published release starts must check which release it is.

This repository publishes releases for more than the platform: the rolling
preview-seed-fixture, and Local PC companion releases to come. A workflow on
``release: published`` that never looks at the tag runs for all of them. The
production deploy did, so any published release would have deployed prod.
"""

import pathlib
import unittest

import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1]
TAG_CHECKS = ("github.event.release.tag_name", "refs/tags/")


def triggers(workflow: dict) -> set[str]:
    """The events a workflow runs on, in any of the forms YAML allows."""
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, (list, dict)):
        return set(on)
    return set()


class ReleaseWorkflowsCheckTheTag(unittest.TestCase):
    def test_every_release_workflow_checks_the_tag(self) -> None:
        unchecked = [
            path.name
            for path in sorted(WORKFLOWS.glob("*.y*ml"))
            if "release" in triggers(yaml.safe_load(text := path.read_text()))
            and not any(check in text for check in TAG_CHECKS)
        ]
        self.assertEqual(
            unchecked,
            [],
            "These workflows run for every published release. Check the tag, "
            "e.g. startsWith(github.event.release.tag_name, 'autogpt-platform-').",
        )

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
        self.assertIn(
            "startsWith(github.event.release.tag_name, 'autogpt-platform-')",
            jobs["migrate"]["if"],
        )
        # Manual dispatch is the only way to deploy outside a platform release.
        self.assertIn("workflow_dispatch", triggers(workflow))
        self.assertIn("github.event_name == 'workflow_dispatch'", jobs["migrate"]["if"])
        # The jobs after it are skipped with it only while they need it and
        # have no condition of their own: `if: always()` on `trigger` would
        # deploy for every published release.
        for job in ("publish-skills", "trigger"):
            self.assertIn("migrate", jobs[job]["needs"])
            self.assertNotIn("if", jobs[job], job)


if __name__ == "__main__":
    unittest.main()
