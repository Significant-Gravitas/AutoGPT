import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRONTEND = "autogpt_platform/frontend/"
# What Vitest, Playwright and Storybook collect, and the files only they use.
TEST_ONLY = re.compile(
    r"(^|/)__tests__/|\.(test|spec|stories)\.[^/]+$|^src/(playwright|tests|mocks)/"
    r"|^coverage/|^(playwright[^/]*\.config\.ts|vitest\.config\.mts"
    r"|test-runner-jest\.config\.js)$"
)
BUILD_INPUTS = (
    "package.json",
    "pnpm-lock.yaml",
    "next.config.mjs",
    "src/app/layout.tsx",
)


def _frontend_images() -> list[Path]:
    """Every Dockerfile that copies the whole frontend into an image."""
    tracked = subprocess.run(
        ["git", "ls-files", "*Dockerfile*"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    copies_frontend = re.compile(rf"^COPY (--\S+ )*{FRONTEND} ", re.M)
    return [
        REPO / path
        for path in tracked
        if not path.endswith(".dockerignore")
        and copies_frontend.search((REPO / path).read_text(encoding="utf-8"))
    ]


def _ignore_file(dockerfile: Path) -> Path:
    # BuildKit prefers an ignore file named after the Dockerfile, next to it.
    own = dockerfile.with_name(f"{dockerfile.name}.dockerignore")
    return own if own.exists() else REPO / ".dockerignore"


def _frontend_context(dockerfile: Path) -> list[str]:
    """The frontend files `dockerfile`'s build gets, as Docker itself filters them."""
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp, "probe.Dockerfile")
        probe.write_text(f"FROM scratch\nCOPY {FRONTEND} /\n", encoding="utf-8")
        shutil.copy(
            _ignore_file(dockerfile), Path(tmp, "probe.Dockerfile.dockerignore")
        )
        out = Path(tmp, "context")
        subprocess.run(
            ["docker", "buildx", "build", "--quiet", "--file", str(probe)]
            + ["--output", f"type=local,dest={out}", str(REPO)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return sorted(
            path.relative_to(out).as_posix()
            for path in out.rglob("*")
            if path.is_file()
        )


@unittest.skipUnless(shutil.which("docker") or os.environ.get("CI"), "needs Docker")
class FrontendBuildContextTests(unittest.TestCase):
    def test_frontend_images_leave_out_the_tests(self):
        images = _frontend_images()
        self.assertGreaterEqual(len(images), 2, "expected the frontend and appliance")
        for dockerfile in images:
            with self.subTest(dockerfile=dockerfile.relative_to(REPO).as_posix()):
                files = _frontend_context(dockerfile)
                for path in BUILD_INPUTS:
                    self.assertIn(path, files)
                leaked = [path for path in files if TEST_ONLY.search(path)]
                if leaked:
                    self.fail(
                        f"{len(leaked)} frontend test files are in the build context "
                        f"of {dockerfile.relative_to(REPO)}, so changing any of them "
                        "rebuilds Next.js from scratch in CI. Exclude them in "
                        f"{_ignore_file(dockerfile).relative_to(REPO)} and in every "
                        "other ignore file that lists frontend tests. First few: "
                        + ", ".join(leaked[:5])
                    )


if __name__ == "__main__":
    unittest.main()
