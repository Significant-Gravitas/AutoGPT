import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        raise SystemExit("Application checks run only in hosted Linux CI")
    frontend = Path.cwd()
    report = Path(os.environ["RUNNER_TEMP"]) / "frontend-gate-canaries"
    report.mkdir()
    probe = frontend / "src/app/ci-gate-negative-probe"
    assert not probe.exists()
    baseline = subprocess.check_output(["git", "diff", "--binary", "--", "."])
    results = []

    def run(name, command, *, rejects=False):
        start = time.monotonic()
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (report / f"{name}.log").write_bytes(result.stdout)
        diagnostic = b"ci-gate-negative-probe" in result.stdout
        passed = (
            (result.returncode == 1 and diagnostic)
            if rejects
            else result.returncode == 0
        )
        results.append(
            {
                "name": name,
                "seconds": time.monotonic() - start,
                "returncode": result.returncode,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"Unexpected result for {name}: {result.returncode}")

    try:
        probe.mkdir()
        page = probe / "page.tsx"
        page.write_text(
            "export default function Page() { return { notAReactNode: true }; }\n"
        )
        run(
            "invalid-route-type",
            [sys.executable, "../../.github/scripts/frontend_typecheck.py"],
            rejects=True,
        )
        page.write_text(
            '/* eslint no-debugger: "error" */\nexport default function Page() { debugger; return null; }\n'
        )
        run("invalid-lint", ["pnpm", "exec", "next", "lint", "--quiet"], rejects=True)
        shutil.rmtree(probe)
        run(
            "restored-types",
            [sys.executable, "../../.github/scripts/frontend_typecheck.py"],
        )
        run("restored-lint", ["pnpm", "exec", "next", "lint", "--quiet"])
        assert (
            subprocess.check_output(["git", "diff", "--binary", "--", "."]) == baseline
        )
    finally:
        if probe.exists():
            shutil.rmtree(probe)
        (report / "results.json").write_text(json.dumps(results, indent=2))
    assert len(results) == 4 and all(r["passed"] for r in results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
