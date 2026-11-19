import os
import re
import subprocess
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "autogpt_platform/frontend"


def main() -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Run the test-runner probes in hosted CI")

    probe = FRONTEND / "src/__ci_vitest_reporting_probe.test.ts"
    with probe.open("x", encoding="utf-8"):
        pass
    cases = {
        "console": """import { it } from "vitest";
it("keeps console output visible", () => {
  console.log("CI_STDOUT_PROBE");
  console.error("CI_STDERR_PROBE");
});
""",
        "rejection": """import { it } from "vitest";
import { setImmediate } from "node:timers/promises";
it("reports an unhandled rejection", async () => {
  void Promise.reject(new Error("CI_UNHANDLED_REJECTION_PROBE"));
  await setImmediate();
});
""",
    }
    try:
        for name, source in cases.items():
            probe.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [
                    "pnpm",
                    "exec",
                    "vitest",
                    "run",
                    str(probe.relative_to(FRONTEND)),
                    "--no-coverage",
                    "--no-cache",
                    "--maxWorkers=1",
                    "--reporter=default",
                ],
                cwd=FRONTEND,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
            if name == "console":
                valid = (
                    result.returncode == 0
                    and "CI_STDOUT_PROBE" in result.stdout
                    and "CI_STDERR_PROBE" in result.stderr
                    and "stdout |" not in output
                    and "stderr |" not in output
                )
            else:
                valid = (
                    result.returncode == 1
                    and "CI_UNHANDLED_REJECTION_PROBE" in output
                    and "Unhandled Rejection" in output
                )
            if not valid:
                raise RuntimeError(f"Vitest {name} reporting probe failed:\n{output}")
            print(f"Vitest {name} reporting probe passed")
    finally:
        probe.unlink()


if __name__ == "__main__":
    main()
