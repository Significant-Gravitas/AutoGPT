import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "autogpt_platform/frontend"


def read_test_inventory(path: Path) -> Counter:
    cases = ElementTree.parse(path).getroot().findall(".//testcase")
    if not cases or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ):
        raise ValueError(f"Missing, failed, or skipped tests in {path}")
    return Counter((case.get("classname"), case.get("name")) for case in cases)


class Benchmark:
    def __init__(self, output: Path):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.canaries = {}
        self.temporary_files = []
        self.setup_diff = subprocess.check_output(
            ["git", "diff", "--binary", "--", "."], cwd=FRONTEND
        )
        (self.output / "setup-changes.patch").write_bytes(self.setup_diff)

    def temporary(self, name: str, content: str) -> Path:
        path = FRONTEND / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
        self.temporary_files.append(path)
        return path

    def run(self, name: str, command: list[str], *, required=False) -> int:
        started = time.perf_counter()
        code = 124
        with (self.output / f"{name}.log").open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=FRONTEND,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                stream.write("\nBenchmark command exceeded 900 seconds.\n")
            except OSError as error:
                code = 127
                stream.write(f"Cannot start benchmark command: {error}\n")
        self.records.append(
            {
                "name": name,
                "seconds": round(time.perf_counter() - started, 3),
                "exit_code": code,
                "command": command,
            }
        )
        print(f"{name}: {self.records[-1]['seconds']}s, exit {code}", flush=True)
        if code and required:
            raise RuntimeError(f"Baseline/setup failed: {name}")
        return code

    def tools(self) -> None:
        self.run("generate-api", ["pnpm", "generate:api"], required=True)
        for name, command in (
            ("eslint", ["pnpm", "exec", "next", "lint", "--no-cache"]),
            ("existing-oxlint", ["pnpm", "lint:shadcn"]),
            ("prettier", ["pnpm", "exec", "prettier", "--check", "."]),
        ):
            self.run(name, command, required=True)
        binaries = Path(os.environ["RUNNER_TEMP"]) / "frontend-tools/node_modules/.bin"
        self.run(
            "install-candidates",
            [
                "npm",
                "install",
                "--prefix",
                str(binaries.parents[1]),
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
                "oxlint@1.90.0",
                "@oxlint/migrate@1.90.0",
                "typescript@7.0.2",
                "oxfmt@0.75.0",
            ],
            required=True,
        )
        self.run(
            "candidate-versions", ["npm", "ls", "--prefix", str(binaries.parents[1])]
        )
        self.run("bun-version", ["bun", "-p", "JSON.stringify(process.versions)"])
        self.typecheck(binaries)
        self.lint(binaries)
        for path in self.temporary_files:
            path.unlink(missing_ok=True)
        # Migration writes only its config; neither formatter is allowed to rewrite source.
        formatter_config = FRONTEND / ".oxfmtrc.json"
        if formatter_config.exists():
            raise RuntimeError("Refusing to overwrite an existing Oxfmt config")
        self.temporary_files.append(formatter_config)
        if (
            self.run(
                "migrate-prettier", [str(binaries / "oxfmt"), "--migrate", "prettier"]
            )
            == 0
        ):
            shutil.copy2(formatter_config, self.output / "oxfmt-config.json")
            self.run("oxfmt", [str(binaries / "oxfmt"), "--check", "."])

    def typecheck(self, binaries: Path) -> None:
        self.run(
            "ts5-effective-config",
            ["pnpm", "exec", "tsc", "--showConfig"],
            required=True,
        )
        self.run(
            "ts5-implicit-types",
            [
                "node",
                "-e",
                'const ts = require("typescript"); console.log(JSON.stringify(ts.getAutomaticTypeDirectiveNames({}, ts.sys)));',
            ],
            required=True,
        )
        options = json.loads((self.output / "ts5-effective-config.log").read_text())[
            "compilerOptions"
        ]
        config = self.temporary(
            "tsconfig.benchmark.json",
            json.dumps(
                {
                    "extends": "./tsconfig.json",
                    "compilerOptions": {
                        "incremental": False,
                        "types": options.get(
                            "types",
                            json.loads(
                                (self.output / "ts5-implicit-types.log").read_text()
                            ),
                        ),
                        "noUncheckedSideEffectImports": options.get(
                            "noUncheckedSideEffectImports", False
                        ),
                        "paths": {
                            name: [
                                path
                                if path.startswith(("./", "../", "/"))
                                else f"./{path}"
                                for path in paths
                            ]
                            for name, paths in options.get("paths", {}).items()
                        },
                    },
                },
                indent=2,
            ),
        )
        shutil.copy2(config, self.output / config.name)
        commands = {
            "tsc5": [
                "pnpm",
                "exec",
                "tsc",
                "--noEmit",
                "--incremental",
                "false",
                "--pretty",
                "false",
                "--extendedDiagnostics",
            ],
            "tsc7": [
                str(binaries / "tsc"),
                "--project",
                config.name,
                "--noEmit",
                "--pretty",
                "false",
                "--extendedDiagnostics",
            ],
            "bun-check": [
                "bun",
                "check",
                "--no-script",
                "--project",
                config.name,
                "--no-pretty",
                "--timing",
            ],
        }
        for name, command in commands.items():
            self.run(name, command, required=name == "tsc5")
        probe = self.temporary(
            "src/__ci_type_probe.ts", 'export const ciMustReject: number = "invalid";\n'
        )
        try:
            for name, command in commands.items():
                code = self.run(f"{name}-negative", command)
                log = (self.output / f"{name}-negative.log").read_text()
                self.canaries[name] = (
                    code in (1, 2) and probe.name in log and "TS2322" in log
                )
        finally:
            probe.unlink()

    def lint(self, binaries: Path) -> None:
        flat = self.temporary(
            "eslint.benchmark.cjs",
            """const { createRequire } = require("node:module");
const eslintRequire = createRequire(require.resolve("eslint"));
const { FlatCompat } = eslintRequire("@eslint/eslintrc");
const js = eslintRequire("@eslint/js");
module.exports = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
}).extends("./.eslintrc.json");
""",
        )
        config = self.temporary(
            ".oxlint.benchmark.json", (FRONTEND / ".oxlintrc.json").read_text()
        )
        migration = self.run(
            "migrate-eslint",
            [
                "node",
                str(binaries.parent / "@oxlint/migrate/dist/bin/oxlint-migrate.mjs"),
                flat.name,
                "--merge",
                "--details",
                "--with-nursery",
                "--output-file",
                config.name,
            ],
        )
        if migration:
            return
        shutil.copy2(config, self.output / "oxlint-config.json")
        candidate = [str(binaries / "oxlint"), "--config", config.name]
        self.run("oxlint-migrated", [*candidate, "src", "scripts"])
        probes = {
            "keyboard": (
                'export function onKey(event: KeyboardEvent) { return event.key === "Enter"; }\n',
                "no-restricted-syntax",
            ),
            "hooks": (
                'import { useState } from "react";\nexport function Probe({ enabled }: { enabled: boolean }) { if (enabled) useState(0); return null; }\n',
                "rules-of-hooks",
            ),
            "compiler": (
                "export function Probe({ value }: { value: { count: number } }) { value.count = 1; return null; }\n",
                "immutability",
            ),
        }
        for name, (source, rule) in probes.items():
            probe = self.temporary(
                f"src/app/(platform)/copilot/__ci_{name}_probe.tsx", source
            )
            try:
                for tool, command in (
                    ("eslint", ["pnpm", "exec", "eslint", "--no-cache"]),
                    ("oxlint", candidate),
                ):
                    label = f"{tool}-negative-{name}"
                    code = self.run(label, [*command, str(probe)])
                    log = (self.output / f"{label}.log").read_text()
                    self.canaries[label] = code == 1 and rule in log
            finally:
                probe.unlink()

    def tests(self, shard: int) -> None:
        workers = len(os.sched_getaffinity(0))
        # Alternate order across shards to expose warm-cache bias in the paired sample.
        layouts = ["isolated", "shared"] if shard % 2 else ["shared", "isolated"]
        for layout in layouts:
            directory = self.output / layout
            directory.mkdir()
            self.run(
                layout,
                [
                    "pnpm",
                    "exec",
                    "vitest",
                    "run",
                    "--coverage",
                    "--coverage.reportOnFailure",
                    f"--shard={shard}/4",
                    "--pool=forks",
                    f"--maxWorkers={workers}",
                    "--no-cache",
                    "--isolate" if layout == "isolated" else "--no-isolate",
                    "--reporter=default",
                    "--reporter=junit",
                    f"--outputFile.junit={directory / 'junit.xml'}",
                    f"--coverage.reportsDirectory={directory}",
                ],
            )
            for validator, report, flags in (
                ("validate_junit.py", "junit.xml", ["--require-no-skips"]),
                ("validate_cobertura.py", "cobertura-coverage.xml", []),
            ):
                self.run(
                    f"{layout}-{validator}",
                    [
                        sys.executable,
                        str(ROOT / ".github/scripts" / validator),
                        *flags,
                        str(directory / report),
                    ],
                )
        try:
            before = read_test_inventory(self.output / "isolated/junit.xml")
            after = read_test_inventory(self.output / "shared/junit.xml")
            self.canaries["same-passing-test-inventory"] = before == after
        except (ValueError, OSError, ElementTree.ParseError):
            self.canaries["same-passing-test-inventory"] = False

    def finish(self) -> None:
        for path in self.temporary_files:
            path.unlink(missing_ok=True)
        final_diff = subprocess.check_output(
            ["git", "diff", "--binary", "--", "."], cwd=FRONTEND
        )
        (self.output / "final-changes.patch").write_bytes(final_diff)
        self.canaries["tracked-files-match-setup"] = final_diff == self.setup_diff
        report = {
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "runner_image": os.environ.get("ImageVersion"),
            "cpu": Path("/proc/cpuinfo")
            .read_text()
            .split("model name", 1)[-1]
            .splitlines()[0],
            "cores": len(os.sched_getaffinity(0)),
            "platform": platform.platform(),
            "lock_sha256": hashlib.sha256(
                (FRONTEND / "pnpm-lock.yaml").read_bytes()
            ).hexdigest(),
            "measurements": self.records,
            "canaries": self.canaries,
            "limitations": [
                "Diagnostic only: success of this workflow does not approve a replacement.",
                "Lint migration warnings and effective rules need review even when canaries pass.",
                "TS7/Bun extend the existing config with TS5 implicit types, side-effect-import default and normalized paths; migration errors are retained.",
                "Test layouts use the same forks pool, core count, coverage and disabled result cache.",
                "A single paired sample cannot establish repeatability or eliminate order-dependent leaks.",
            ],
        }
        (self.output / "measurements.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write(
                    "Diagnostic benchmark; replacements remain disabled.\n\n| Command | Seconds | Exit |\n|---|---:|---:|\n"
                )
                for row in self.records:
                    stream.write(
                        f"| {row['name']} | {row['seconds']} | {row['exit_code']} |\n"
                    )
                stream.write(
                    f"\nCompatibility canaries: `{json.dumps(self.canaries)}`\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tools", "tests"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(1, 5))
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
        parser.error("Dependency benchmarks run only on hosted Linux CI")
    if args.mode == "tests" and args.shard is None:
        parser.error("tests requires --shard")
    benchmark = Benchmark(args.output)
    try:
        if args.mode == "tools":
            benchmark.tools()
        else:
            benchmark.tests(args.shard)
    finally:
        benchmark.finish()


if __name__ == "__main__":
    main()
