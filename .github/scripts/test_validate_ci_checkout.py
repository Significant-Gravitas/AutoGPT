import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("validate_ci_checkout.py")


class CheckoutContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "--quiet")
        files = {
            "autogpt_platform/backend/backend/server.py": "source",
            "autogpt_platform/backend/backend/server_test.py": "test",
            "autogpt_platform/frontend/src/page.test.ts": "test",
            "docs/platform/.gitbook/assets/image.png": "image",
            "docs/platform/.gitbook/assets/demo.gif": "image",
            "docs/platform/.gitbook/assets/README.md": "documentation",
            "docs/platform/guide.md": "documentation",
            "autogpt_platform/backend/test/fixture.png": "fixture",
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Checkout fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={self.root / 'empty-hooks'}",
            "commit",
            "--quiet",
            "-m",
            "ci: create checkout fixture",
        )

    def git(self, *args, input=None):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            input=input,
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )

    def validate(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def test_full_checkout_passes(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("omitted 0 documentation images", result.stdout)

    def test_real_sparse_checkout_retains_all_other_tracked_files(self):
        self.git(
            "sparse-checkout",
            "set",
            "--no-cone",
            "--stdin",
            input="/*\n!/docs/platform/.gitbook/assets/*.png\n!/docs/platform/.gitbook/assets/*.gif\n",
        )
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("omitted 2 documentation images", result.stdout)
        for name in [
            "docs/platform/.gitbook/assets/README.md",
            "autogpt_platform/backend/test/fixture.png",
        ]:
            self.assertTrue((self.root / name).is_file(), name)

    def test_missing_test_source_fails(self):
        name = "autogpt_platform/backend/backend/server_test.py"
        (self.root / name).unlink()
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(name, result.stderr)

    def test_missing_non_image_documentation_fails(self):
        name = "docs/platform/.gitbook/assets/README.md"
        (self.root / name).unlink()
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(name, result.stderr)

    def test_missing_image_test_fixture_fails(self):
        name = "autogpt_platform/backend/test/fixture.png"
        (self.root / name).unlink()
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(name, result.stderr)


if __name__ == "__main__":
    unittest.main()
