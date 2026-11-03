"""Every workflow a published release starts must check which release it is.

This repository publishes releases for more than the platform: the rolling
preview-seed-fixture, and Local PC companion releases to come. A workflow on
``release: published`` that never looks at the tag runs for all of them. The
production deploy did, so any published release would have deployed prod.
"""

import pathlib
import re
import unittest

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1]
RELEASE_TRIGGER = re.compile(r"^  release:\s*$", re.MULTILINE)
TAG_CHECKS = ("github.event.release.tag_name", "refs/tags/")


class ReleaseWorkflowsCheckTheTag(unittest.TestCase):
    def test_every_release_workflow_checks_the_tag(self) -> None:
        unchecked = [
            path.name
            for path in sorted(WORKFLOWS.glob("*.y*ml"))
            if RELEASE_TRIGGER.search(text := path.read_text())
            and not any(check in text for check in TAG_CHECKS)
        ]
        self.assertEqual(
            unchecked,
            [],
            "These workflows run for every published release. Check the tag, "
            "e.g. startsWith(github.event.release.tag_name, 'autogpt-platform-').",
        )

    def test_production_deploys_only_platform_releases(self) -> None:
        text = (WORKFLOWS / "platform-autogpt-deploy-prod.yml").read_text()
        self.assertIn(
            "startsWith(github.event.release.tag_name, 'autogpt-platform-')", text
        )


if __name__ == "__main__":
    unittest.main()
