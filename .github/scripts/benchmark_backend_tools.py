import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "inventory", "compare"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.mode == "inventory":
        print(json.dumps(inventory(), indent=2, sort_keys=True))
    elif args.mode == "compare":
        if not args.baseline or not args.candidate:
            parser.error("compare requires --baseline and --candidate")
        differences = compare_inventories(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
        )
        print(json.dumps(differences, indent=2, sort_keys=True))
        sys.exit(bool(differences))
    else:
        if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux":
            parser.error("Dependency benchmarks run only on hosted Linux CI")
        if not args.output:
            parser.error("run requires --output")
        benchmark = Benchmark(args.output.resolve())
        try:
            benchmark.install()
            benchmark.check_tools()
        finally:
            benchmark.finish()


def inventory() -> dict:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        packages[name] = {
            "version": distribution.version,
            "url": direct.get("url"),
            "commit": direct.get("vcs_info", {}).get("commit_id"),
            "editable": direct.get("dir_info", {}).get("editable", False),
        }
    return packages


def compare_inventories(baseline: dict, candidate: dict) -> dict:
    return {
        name: {"baseline": baseline.get(name), "candidate": candidate.get(name)}
        for name in sorted(baseline.keys() | candidate.keys())
        if name != "pip" and baseline.get(name) != candidate.get(name)
    }


class Benchmark:
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.script = Path(__file__).resolve()
        self.root = self.script.parents[2]
        self.backend = self.root / "autogpt_platform/backend"
        self.uv_env = Path(os.environ["RUNNER_TEMP"]) / "backend-uv-benchmark"
        self.records: list[dict] = []

    def run(
        self,
        name: str,
        command: list[str],
        *,
        required: bool = True,
        environment: dict[str, str] | None = None,
    ) -> int:
        started = time.perf_counter()
        code = 124
        log = self.output / f"{name}.log"
        try:
            with log.open("w", encoding="utf-8") as stream:
                code = subprocess.run(
                    command,
                    cwd=self.backend,
                    env={**os.environ, **(environment or {})},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                ).returncode
        finally:
            self.records.append(
                {
                    "name": name,
                    "seconds": round(time.perf_counter() - started, 3),
                    "exit_code": code,
                    "required": required,
                    "command": command,
                }
            )
            print(f"{name}: {self.records[-1]['seconds']}s, exit {code}", flush=True)
        if code and required:
            raise RuntimeError(f"{name} failed; see {log.name}")
        return code

    def install(self) -> None:
        self.run("poetry-install-cold", ["poetry", "install", "--no-interaction"])
        self.run("poetry-install-warm", ["poetry", "install", "--no-interaction"])
        try:
            self.install_uv()
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"uv installation benchmark incomplete: {error}", flush=True)
        self.run(
            "lock-unchanged",
            ["git", "diff", "--exit-code", "--", "poetry.lock", "pyproject.toml"],
        )

    def install_uv(self) -> None:
        project = tomllib.loads((self.backend / "pyproject.toml").read_text())
        poetry = project["tool"]["poetry"]["dependencies"]["poetry"]
        requirements = self.output / "requirements.txt"
        self.run(
            "poetry-lock-export",
            [
                "uv",
                "tool",
                "run",
                "--from",
                f"poetry=={poetry}",
                "--with",
                "poetry-plugin-export==1.10.1",
                "poetry",
                "export",
                "--with",
                "dev",
                "--format",
                "requirements.txt",
                "--output",
                str(requirements),
            ],
            environment={
                "UV_CACHE_DIR": str(Path(os.environ["RUNNER_TEMP"]) / "uv-export-cache")
            },
        )
        self.run(
            "uv-venv", ["uv", "venv", "--python", sys.executable, str(self.uv_env)]
        )
        target = str(self.uv_env / "bin/python")
        activated = {
            "VIRTUAL_ENV": str(self.uv_env),
            "POETRY_VIRTUALENVS_IN_PROJECT": "false",
            "PATH": f"{self.uv_env / 'bin'}{os.pathsep}{os.environ['PATH']}",
        }
        for temperature in ("cold", "warm"):
            self.run(
                f"uv-sync-{temperature}",
                ["uv", "pip", "sync", "--python", target, str(requirements)],
            )
            self.run(
                f"uv-poetry-target-{temperature}",
                ["poetry", "env", "info", "--path"],
                environment=activated,
            )
            selected = (
                (self.output / f"uv-poetry-target-{temperature}.log")
                .read_text()
                .strip()
            )
            if Path(selected).resolve() != self.uv_env.resolve():
                raise RuntimeError(
                    "Poetry selected the wrong environment for root registration"
                )
            self.run(
                f"uv-root-{temperature}",
                ["poetry", "install", "--only-root", "--no-interaction"],
                environment=activated,
            )
        for name, python in (
            ("poetry", self.backend / ".venv/bin/python"),
            ("uv", self.uv_env / "bin/python"),
        ):
            self.run(f"{name}-inventory", [str(python), str(self.script), "inventory"])
            self.run(
                f"{name}-dependency-check",
                ["uv", "pip", "check", "--python", str(python)],
                required=False,
            )
            self.run(
                f"{name}-shard-helper-tests",
                [
                    str(python),
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(self.root / ".github/scripts"),
                    "-p",
                    "test_backend_test_shard.py",
                ],
            )
        self.run(
            "compare-environments",
            [
                sys.executable,
                str(self.script),
                "compare",
                "--baseline",
                str(self.output / "poetry-inventory.log"),
                "--candidate",
                str(self.output / "uv-inventory.log"),
            ],
            required=False,
        )

    def check_tools(self) -> None:
        files = (
            subprocess.check_output(
                ["git", "ls-files", "-z", "--", "*.py", "*.pyi"],
                cwd=self.backend,
            )
            .decode()
            .split("\0")
        )
        files = sorted(
            path for path in files if path and not path.startswith("test/fixtures/")
        )
        (self.output / "formatter-files.json").write_text(json.dumps(files, indent=2))
        for name, command in (
            ("black-check", ["black", "--check", "--diff"]),
            (
                "ruff-format-check",
                ["ruff", "format", "--check", "--diff", "--no-cache"],
            ),
            ("isort-check", ["isort", "--check", "--diff", "--profile", "black"]),
            ("ruff-import-check", ["ruff", "check", "--select", "I", "--no-cache"]),
        ):
            self.run(name, ["poetry", "run", *command, *files], required=False)
        self.run("prisma-generate", ["poetry", "run", "prisma", "generate"])
        self.run("prisma-stubs", ["poetry", "run", "gen-prisma-stub"])
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.run(
            "pyright",
            ["poetry", "run", "pyright", "--pythonversion", version],
            required=False,
        )
        self.run(
            "ty-bootstrap",
            ["uv", "tool", "run", "--from", "ty==0.0.95", "ty", "--version"],
        )
        config = json.loads((self.backend / "pyrightconfig.json").read_text())
        # Pyright's ignored files are still analyzed for imports. Suppress only
        # their ty diagnostics too, rather than excluding them from discovery.
        ty_config = self.backend / "ty.benchmark.toml"
        ty_config.write_text(
            "[src]\ninclude = "
            + json.dumps(config["include"])
            + "\n[[overrides]]\ninclude = "
            + json.dumps(config["ignore"])
            + '\n[overrides.rules]\nall = "ignore"\n',
            encoding="utf-8",
        )
        (self.output / ty_config.name).write_bytes(ty_config.read_bytes())
        try:
            self.run(
                "ty",
                [
                    "uv",
                    "tool",
                    "run",
                    "--from",
                    "ty==0.0.95",
                    "ty",
                    "check",
                    "--config-file",
                    str(ty_config),
                    "--python",
                    str(self.backend / ".venv"),
                    "--python-version",
                    version,
                    "--output-format",
                    "concise",
                ],
                required=False,
            )
        finally:
            ty_config.unlink()

    def finish(self) -> None:
        report = {
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root
            )
            .decode()
            .strip(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu_model": next(
                (
                    line.partition(":")[2].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "runner_image": os.environ.get("ImageVersion"),
            "poetry_lock_sha256": hashlib.sha256(
                (self.backend / "poetry.lock").read_bytes()
            ).hexdigest(),
            "measurements": self.records,
            "limitations": [
                "One paired sample per Python version; no full backend runtime validation.",
                "Export uses a separate download cache; include export cost in uv install totals.",
                "Warm install reuses the installed environment, not only the download cache.",
                "Poetry and uv bootstrap Actions steps are outside command timings.",
                "Formatter outputs and type-checker rules can differ; timings do not establish equivalence.",
                "Only pip, the virtualenv seed package, is ignored when comparing inventories.",
            ],
        }
        (self.output / "measurements.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        lines = [
            "### Tool benchmark (diagnostic only)",
            "",
            "| Command | Seconds | Exit |",
            "|---|---:|---:|",
        ]
        lines.extend(
            f"| {r['name']} | {r['seconds']} | {r['exit_code']} |" for r in self.records
        )
        lines.extend(
            [
                "",
                "A successful benchmark job means reports were collected, not that a replacement passed.",
            ]
        )
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary).open("a", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
