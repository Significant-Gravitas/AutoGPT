from __future__ import annotations

import re
import unittest
from pathlib import Path, PurePosixPath

ASSET_DIR = Path(__file__).resolve().parents[1]
APPLIANCE_DOCKERFILE = ASSET_DIR / "Dockerfile"
BACKEND_DOCKERFILE = ASSET_DIR.parent / "backend" / "Dockerfile"
BAKE_PATH = ASSET_DIR / "docker-bake.hcl"

SERVER_BASE = "FROM debian:13-slim AS server-base"
SERVER = "FROM server-base AS server"
APPLIANCE = "FROM autogpt-backend-base AS single-container"
LAYER_STEPS = {"ADD", "COPY", "RUN"}


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


def run_commands(lines: list[str]) -> set[str]:
    return {
        command.strip()
        for line in lines
        if line.startswith("RUN ")
        for command in line.removeprefix("RUN ").split("&&")
    }


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
            i for i, line in enumerate(self.appliance) if line.startswith("ARG ")
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


if __name__ == "__main__":
    unittest.main()
