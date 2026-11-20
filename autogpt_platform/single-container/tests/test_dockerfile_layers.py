from __future__ import annotations

import re
import unittest
from pathlib import Path, PurePosixPath

ASSET_DIR = Path(__file__).resolve().parents[1]
APPLIANCE_DOCKERFILE = ASSET_DIR / "Dockerfile"
BACKEND_DOCKERFILE = ASSET_DIR.parent / "backend" / "Dockerfile"
BAKE_PATH = ASSET_DIR / "docker-bake.hcl"
WORKFLOW_PATH = (
    ASSET_DIR.parents[1] / ".github/workflows/platform-single-container-docker.yml"
)

SERVER_BASE = "FROM debian:13-slim AS server-base"
SERVER = "FROM server-base AS server"
APPLIANCE = "FROM autogpt-backend-base AS single-container"
LAYER_STEPS = {"ADD", "COPY", "RUN"}
PACKAGE_REFRESH = "APT_REFRESH"
PACKAGE_INSTALL = re.compile(r"\bapt(?:-get)? (?:update|install|upgrade)\b")


def instructions(path: Path) -> list[str]:
    """Return a Dockerfile's instructions with continuation lines joined."""
    result: list[str] = []
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}" if current else stripped
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        result.append(re.sub(r"\s+", " ", current))
        current = ""
    return result


def stage(lines: list[str], header: str) -> list[str]:
    start = lines.index(header) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("FROM ")),
        len(lines),
    )
    return lines[start:end]


def stages(lines: list[str]) -> dict[str, list[str]]:
    """Map each stage's FROM instruction to the instructions in that stage."""
    result: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        if line.startswith("FROM "):
            current = result.setdefault(line, [])
        elif current is not None:
            current.append(line)
    return result


def run_commands(lines: list[str]) -> set[str]:
    return {
        command.strip()
        for line in lines
        if line.startswith("RUN ")
        for command in line.removeprefix("RUN ").split("&&")
    }


def declares(line: str, argument: str) -> bool:
    return line.startswith("ARG ") and line.split()[1].split("=")[0] == argument


def installs_packages(line: str) -> bool:
    return line.startswith("RUN ") and bool(PACKAGE_INSTALL.search(line))


def workflow_jobs(workflow: str) -> dict[str, str]:
    """Map each job id in a workflow to its indented body."""
    jobs = workflow.split("\njobs:\n", 1)[1]
    return dict(re.findall(r"^  ([\w-]+):\n((?:(?:    .*)?\n)*)", jobs, re.M))


class ApplianceLayerOrderTest(unittest.TestCase):
    """The appliance stacks its dependencies under AutoGPT's code.

    It builds on the backend's server-base target instead of the finished
    server image, so the layers that only change with dependencies are reused
    by every build, and a new commit only re-stores the code layers on top.
    """

    def setUp(self) -> None:
        self.backend = instructions(BACKEND_DOCKERFILE)
        self.appliance = stage(instructions(APPLIANCE_DOCKERFILE), APPLIANCE)

    def test_server_base_holds_dependencies_only(self) -> None:
        for line in stage(self.backend, SERVER_BASE):
            if line.startswith(("COPY ", "ADD ")):
                self.assertTrue(
                    line.startswith("COPY --from=builder "),
                    f"server-base must not copy AutoGPT code: {line}",
                )

    def test_appliance_adds_everything_the_server_target_adds(self) -> None:
        copied_from_backend = {
            parts[-1]
            for parts in (line.split() for line in self.appliance)
            if parts[:2] == ["COPY", "--from=backend-code"] and parts[2] == parts[-1]
        }
        commands = run_commands(self.appliance)

        workdir = PurePosixPath("/")
        for line in stage(self.backend, SERVER_BASE):
            if line.startswith("WORKDIR "):
                workdir = workdir / line.split()[1]
        for line in stage(self.backend, SERVER):
            keyword, *arguments = line.split()
            if keyword == "WORKDIR":
                workdir = workdir / arguments[0]
            elif keyword in {"COPY", "ADD"}:
                destination = str(workdir / arguments[-1])
                self.assertTrue(
                    any(
                        destination == copied or destination.startswith(f"{copied}/")
                        for copied in copied_from_backend
                    ),
                    f"the appliance never copies {destination} from backend-code",
                )
            elif keyword == "RUN":
                self.assertIn(line.removeprefix("RUN "), commands)

    def test_build_arguments_follow_every_layer(self) -> None:
        arguments = [
            i
            for i, line in enumerate(self.appliance)
            if line.startswith("ARG ") and not declares(line, PACKAGE_REFRESH)
        ]
        self.assertTrue(arguments, "the appliance declares its build arguments")
        self.assertEqual(
            [
                line
                for line in self.appliance[min(arguments) :]
                if line.split()[0] in LAYER_STEPS
            ],
            [],
            "build arguments change every build and would invalidate later layers",
        )

    def test_bake_supplies_server_base(self) -> None:
        bake = BAKE_PATH.read_text(encoding="utf-8")
        self.assertRegex(
            bake, r'target "backend-server-base" \{[^}]*target\s*=\s*"server-base"'
        )
        self.assertRegex(
            bake, r'autogpt-backend-base\s*=\s*"target:backend-server-base"'
        )


class PackageRefreshTest(unittest.TestCase):
    """OS package installs run again once a day instead of staying cached.

    A build argument is part of the cache key of every RUN after its ARG. Each
    stage that installs packages declares APT_REFRESH before its first install
    and CI passes the UTC date, so Debian's security fixes reach the images
    within a day rather than waiting behind a cached layer.
    """

    def test_every_package_install_follows_the_refresh_argument(self) -> None:
        for dockerfile in (BACKEND_DOCKERFILE, APPLIANCE_DOCKERFILE):
            installing = {
                header: body
                for header, body in stages(instructions(dockerfile)).items()
                if any(installs_packages(line) for line in body)
            }
            self.assertTrue(installing, f"no package installs found in {dockerfile}")
            for header, body in installing.items():
                first_install = next(
                    i for i, line in enumerate(body) if installs_packages(line)
                )
                with self.subTest(dockerfile=dockerfile.parent.name, stage=header):
                    self.assertTrue(
                        any(
                            declares(line, PACKAGE_REFRESH)
                            for line in body[:first_install]
                        ),
                        f"{PACKAGE_REFRESH} must be declared before the first install",
                    )

    def test_chromium_install_follows_the_refresh_argument(self) -> None:
        installs = [
            body[:i]
            for body in stages(instructions(BACKEND_DOCKERFILE)).values()
            for i, line in enumerate(body)
            if installs_packages(line) and re.search(r"\bchromium\b", line)
        ]
        self.assertEqual(len(installs), 1, "the backend installs Chromium once")
        self.assertTrue(
            any(declares(line, PACKAGE_REFRESH) for line in installs[0]),
            "a cached Chromium layer would keep shipping a vulnerable browser",
        )

    def test_every_image_build_passes_the_utc_date(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        builds = 0
        for job, body in workflow_jobs(workflow).items():
            steps = re.split(r"\n      - ", body)
            for index, step in enumerate(steps):
                if "uses: docker/bake-action@" not in step:
                    continue
                builds += 1
                with self.subTest(job=job):
                    self.assertIn(
                        f"*.args.{PACKAGE_REFRESH}="
                        "${{ steps.apt-refresh.outputs.date }}",
                        step,
                    )
                    dates = [
                        earlier
                        for earlier in steps[:index]
                        if re.search(r"^\s+id: apt-refresh$", earlier, re.M)
                    ]
                    self.assertEqual(len(dates), 1, "an earlier step sets the date")
                    self.assertIn(
                        'echo "date=$(date -u +%F)" >> "$GITHUB_OUTPUT"', dates[0]
                    )
        self.assertEqual(builds, workflow.count("uses: docker/bake-action@"))


if __name__ == "__main__":
    unittest.main()
