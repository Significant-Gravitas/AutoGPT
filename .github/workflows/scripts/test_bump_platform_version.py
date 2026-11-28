"""The release version bump changes only the two version lines, and only forward."""

import json
import os
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock

import yaml

import bump_platform_version as bump

REPO = pathlib.Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github/workflows/platform-version-bump.yml"

PYPROJECT = """\
[tool.poetry]
name = "autogpt-platform-backend"
version = "0.6.22"
description = "A platform"
packages = [{ include = "backend", format = "sdist" }]

[tool.poetry.dependencies]
python = ">=3.10,<3.14"
httpx = { extras = ["socks"], version = "^0.28.1" }
"""

PACKAGE_JSON = """\
{
  "name": "frontend",
  "version": "0.3.4",
  "private": true,
  "dependencies": {
    "some-package": {
      "version": "1.0.0"
    }
  }
}
"""


class VersionFromTag(unittest.TestCase):
    def test_platform_release_tags(self) -> None:
        self.assertEqual(
            bump.version_from_tag("autogpt-platform-beta-v0.12.1"), "0.12.1"
        )
        self.assertEqual(bump.version_from_tag("autogpt-platform-v1.0.0"), "1.0.0")

    def test_other_tags_are_refused(self) -> None:
        for tag in (
            "v0.12.1",
            "preview-seed-fixture",
            "autogpt-platform-beta-v0.12",
            "autogpt-platform-beta-v0.12.1-rc1",
            "autogpt-platform-beta-v0.12.1\n",
            "local-executor-v1.0.0",
        ):
            with self.assertRaises(bump.BumpError, msg=tag):
                bump.version_from_tag(tag)


class NewestVersion(unittest.TestCase):
    TAGS = [
        "autogpt-platform-beta-v0.11.3",
        "autogpt-platform-beta-v0.12.0",
        "autogpt-platform-beta-v0.12.1",
        "preview-seed-fixture",
        "v99.0.0",
    ]

    def test_the_newest_release_keeps_its_version(self) -> None:
        self.assertEqual(bump.newest_version("0.12.1", self.TAGS), "0.12.1")
        self.assertEqual(bump.newest_version("0.13.0", self.TAGS), "0.13.0")
        self.assertEqual(bump.newest_version("0.13.0", []), "0.13.0")

    def test_an_older_lines_hotfix_sets_the_newest_release(self) -> None:
        # Other products' tags and unprefixed ones (v99.0.0) don't count.
        self.assertEqual(bump.newest_version("0.11.4", self.TAGS), "0.12.1")


class BumpText(unittest.TestCase):
    def test_pyproject_changes_only_the_poetry_version(self) -> None:
        text, changed = bump.bump_pyproject(PYPROJECT, "0.12.1")
        self.assertTrue(changed)
        self.assertEqual(
            text, PYPROJECT.replace('version = "0.6.22"', 'version = "0.12.1"')
        )
        self.assertIn('version = "^0.28.1"', text)

    def test_package_json_changes_only_the_top_level_version(self) -> None:
        text, changed = bump.bump_package_json(PACKAGE_JSON, "0.12.1")
        self.assertTrue(changed)
        self.assertEqual(
            text, PACKAGE_JSON.replace('"version": "0.3.4"', '"version": "0.12.1"')
        )
        self.assertEqual(
            json.loads(text)["dependencies"]["some-package"]["version"], "1.0.0"
        )

    def test_versions_compare_as_numbers(self) -> None:
        # 0.12.1 is newer than 0.6.22, though it sorts before it as text.
        self.assertTrue(bump.bump_pyproject(PYPROJECT, "0.12.1")[1])
        self.assertTrue(bump.bump_pyproject(PYPROJECT, "0.6.23")[1])

    def test_same_or_older_version_is_left_alone(self) -> None:
        for version in ("0.6.22", "0.6.21", "0.5.99"):
            self.assertEqual(
                bump.bump_pyproject(PYPROJECT, version), (PYPROJECT, False)
            )
        for version in ("0.3.4", "0.2.0"):
            self.assertEqual(
                bump.bump_package_json(PACKAGE_JSON, version), (PACKAGE_JSON, False)
            )

    def test_a_missing_version_line_fails(self) -> None:
        with self.assertRaises(bump.BumpError):
            bump.bump_pyproject(PYPROJECT.replace('version = "0.6.22"\n', ""), "0.12.1")
        with self.assertRaises(bump.BumpError):
            bump.bump_package_json(
                PACKAGE_JSON.replace('  "version": "0.3.4",\n', ""), "0.12.1"
            )

    def test_a_current_version_that_is_not_numeric_fails(self) -> None:
        with self.assertRaises(bump.BumpError):
            bump.bump_pyproject(PYPROJECT.replace("0.6.22", "0.6.22a1"), "0.12.1")


class BumpRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        for path in (bump.PYPROJECT, bump.PACKAGE_JSON):
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / path, self.root / path)

    def test_the_real_files_bump(self) -> None:
        # The repository's own files have the lines the bump looks for.
        output = self.root / "github_output"
        env = {**os.environ, "GITHUB_OUTPUT": str(output)}
        with unittest.mock.patch.dict(os.environ, env):
            code = bump.main(
                ["autogpt-platform-beta-v999.0.0", "--root", str(self.root)]
            )
        self.assertEqual(code, 0)
        pyproject = (self.root / bump.PYPROJECT).read_text()
        package = json.loads((self.root / bump.PACKAGE_JSON).read_text())
        self.assertIn('\nversion = "999.0.0"\n', pyproject)
        self.assertEqual(package["version"], "999.0.0")
        self.assertEqual(output.read_text(), "version=999.0.0\nchanged=true\n")
        # Exactly one line changed in each file.
        for path in (bump.PYPROJECT, bump.PACKAGE_JSON):
            before = (REPO / path).read_text().splitlines()
            after = (self.root / path).read_text().splitlines()
            self.assertEqual(len(before), len(after), path)
            self.assertEqual(sum(a != b for a, b in zip(before, after)), 1, path)

    def test_nothing_to_bump(self) -> None:
        output = self.root / "github_output"
        with unittest.mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
            code = bump.main(["autogpt-platform-beta-v0.0.1", "--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(output.read_text(), "version=0.0.1\nchanged=false\n")
        for path in (bump.PYPROJECT, bump.PACKAGE_JSON):
            self.assertEqual((self.root / path).read_text(), (REPO / path).read_text())

    def test_a_hotfix_older_than_the_newest_release_sets_the_newest(self) -> None:
        # dev still has the old versions, but a newer release exists, so its
        # pending bump must not be replaced with the hotfix's version.
        tags = self.root / "tags"
        tags.write_text(
            "autogpt-platform-beta-v999.1.0\nautogpt-platform-beta-v999.0.1\n"
        )
        output = self.root / "github_output"
        with unittest.mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
            code = bump.main(
                [
                    "autogpt-platform-beta-v999.0.1",
                    "--root",
                    str(self.root),
                    "--tags",
                    str(tags),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(output.read_text(), "version=999.1.0\nchanged=true\n")
        package = json.loads((self.root / bump.PACKAGE_JSON).read_text())
        self.assertEqual(package["version"], "999.1.0")

    def test_an_unreadable_file_fails_the_run(self) -> None:
        (self.root / bump.PACKAGE_JSON).write_text('{\n  "version": "0.3.4",\n')
        tag = "autogpt-platform-beta-v999.0.0"
        self.assertEqual(bump.main([tag, "--root", str(self.root)]), 1)

    def test_a_bad_tag_fails_the_run(self) -> None:
        self.assertEqual(bump.main(["v0.12.1", "--root", str(self.root)]), 1)


class Workflow(unittest.TestCase):
    """The workflow runs on platform releases only and passes no expression into a script."""

    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.job = self.workflow["jobs"]["bump"]
        self.steps = {step.get("id"): step for step in self.job["steps"]}

    def test_runs_on_published_platform_releases_only(self) -> None:
        on = self.workflow.get("on", self.workflow.get(True))
        self.assertEqual(on, {"release": {"types": ["published"]}})
        self.assertEqual(
            self.job["if"],
            "startsWith(github.event.release.tag_name, 'autogpt-platform-') && "
            "!github.event.release.prerelease",
        )

    def test_the_tag_reaches_the_script_through_the_environment(self) -> None:
        step = self.steps["bump"]
        self.assertNotIn("${{", step["run"])
        self.assertEqual(step["env"]["TAG"], "${{ github.event.release.tag_name }}")

    def test_only_published_stable_releases_are_candidates(self) -> None:
        # A pushed tag, draft or prerelease ahead of the stable line must not
        # put an unreleased version on dev.
        run = self.steps["bump"]["run"]
        self.assertNotIn("ls-remote", run)
        self.assertIn("/releases", run)
        self.assertIn("--paginate", run)
        self.assertIn("select(.draft == false and .prerelease == false)", run)

    def test_the_pr_goes_to_dev_with_auto_merge(self) -> None:
        pr = self.steps["pr"]["with"]
        self.assertEqual(pr["base"], "dev")
        self.assertEqual(
            set(pr["add-paths"].split()), {str(bump.PYPROJECT), str(bump.PACKAGE_JSON)}
        )
        self.assertIn("gh pr merge", self.steps["auto-merge"]["run"])
        self.assertIn("--auto", self.steps["auto-merge"]["run"])
        # No prompt for a merge method in a run without a terminal.
        self.assertIn("--squash", self.steps["auto-merge"]["run"])
        # Opened as the App: CI doesn't run on a PR opened with GITHUB_TOKEN.
        self.assertEqual(pr["token"], "${{ steps.app-token.outputs.token }}")
        self.assertEqual(self.workflow["permissions"], {})
        self.assertEqual(self.job["permissions"], {"contents": "read"})
        token = self.steps["app-token"]["with"]
        self.assertEqual(token["permission-contents"], "write")
        self.assertEqual(token["permission-pull-requests"], "write")
        self.assertIs(pr["sign-commits"], True)
        auto_merge = self.steps["auto-merge"]
        self.assertEqual(
            auto_merge["env"]["GH_TOKEN"], "${{ steps.app-token.outputs.token }}"
        )
        self.assertIn("steps.pr.outputs.pull-request-number", auto_merge["if"])

    def test_concurrency_is_on_the_job(self) -> None:
        # A workflow-level group would let a skipped run for another
        # product's release replace a pending platform run.
        self.assertNotIn("concurrency", self.workflow)
        self.assertEqual(self.job["concurrency"]["cancel-in-progress"], False)


if __name__ == "__main__":
    unittest.main()
