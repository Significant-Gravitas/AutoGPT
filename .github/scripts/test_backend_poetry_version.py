import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class BackendPoetryVersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.source = root / "source"
        self.checkout = root / "shallow checkout"
        self.source.mkdir()
        self.git(self.source, "init", "-q")
        self.git(self.source, "config", "user.email", "ci@example.invalid")
        self.git(self.source, "config", "user.name", "CI fixture")
        repo = Path(__file__).resolve().parents[2]
        helper = Path(".github/workflows/scripts/get_package_version_from_lockfile.py")
        (self.source / helper).parent.mkdir(parents=True)
        shutil.copyfile(repo / helper, self.source / helper)
        self.lock = self.source / "autogpt_platform/backend/poetry.lock"
        self.lock.parent.mkdir(parents=True)
        workflow = yaml.safe_load(
            (repo / ".github/workflows/platform-backend-ci.yml").read_text()
        )
        step = next(
            step
            for step in workflow["jobs"]["test"]["steps"]
            if step.get("name") == "Install Poetry"
        )
        self.select_version = step["run"].split("# Install Poetry", 1)[0]

    def git(self, directory, *args):
        return subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-C", str(directory), *args],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def revision(self, version):
        self.lock.write_text(f'[[package]]\nname = "poetry"\nversion = "{version}"\n')
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-qm", "Update fixture")
        return self.git(self.source, "rev-parse", "HEAD")

    def clone(self):
        self.git(
            self.source,
            "clone",
            "-q",
            "--depth=1",
            self.source.as_uri(),
            str(self.checkout),
        )
        self.assertEqual(self.git(self.checkout, "rev-list", "--count", "HEAD"), "1")

    def select(self, base_sha, base_ref="refs/heads/dev"):
        script = Path(self.directory.name) / "select-version.sh"
        script.write_text(self.select_version, encoding="utf-8")
        return subprocess.run(
            [shutil.which("bash") or "bash", "-e", "-o", "pipefail", script.as_posix()],
            cwd=self.checkout / "autogpt_platform/backend",
            env={
                **os.environ,
                "BASE_REF": base_ref,
                "BASE_SHA": base_sha,
                "PATH": str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ["PATH"],
            },
            capture_output=True,
            text=True,
        )

    def test_reads_pinned_base_after_branch_moves_and_keeps_newer_version(self):
        base = self.revision("2.3.0")
        self.revision("2.2.1")
        self.clone()
        result = self.select(base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Using Poetry version 2.3.0", result.stdout)

    def test_keeps_newer_head_and_supports_events_without_a_base(self):
        base = self.revision("2.1.0")
        self.revision("2.2.1")
        self.clone()
        for sha, ref in ((base, "refs/heads/dev"), ("", "")):
            with self.subTest(base=sha):
                result = self.select(sha, ref)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Using Poetry version 2.2.1", result.stdout)

    def test_rejects_missing_invalid_and_unavailable_base_commits(self):
        self.revision("2.2.1")
        self.clone()
        for base in ("", "--upload-pack=unexpected", "f" * 40):
            with self.subTest(base=base):
                result = self.select(base)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Using Poetry version", result.stdout)


if __name__ == "__main__":
    unittest.main()
