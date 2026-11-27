from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path, PurePosixPath

ASSET_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASSET_DIR.parents[1]
APPLIANCE_DOCKERFILE = ASSET_DIR / "Dockerfile"
BACKEND_DOCKERFILE = ASSET_DIR.parent / "backend" / "Dockerfile"
BAKE_PATH = ASSET_DIR / "docker-bake.hcl"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/platform-single-container-docker.yml"
REFRESH_SCRIPT = REPO_ROOT / ".github/scripts/apt_refresh.py"
SPEC = importlib.util.spec_from_file_location("apt_refresh", REFRESH_SCRIPT)
apt_refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(apt_refresh)

SERVER_BASE = "FROM debian:13-slim AS server-base"
SERVER = "FROM server-base AS server"
APPLIANCE = "FROM autogpt-backend-base AS single-container"
LAYER_STEPS = {"ADD", "COPY", "RUN"}
PACKAGE_REFRESH = "APT_REFRESH"
PACKAGE_INSTALL = re.compile(r"\bapt(?:-get)? (?:update|install|upgrade)\b")
DEBIAN_CODENAMES = {"12": "bookworm", "13": "trixie", "14": "forky", "15": "duke"}


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


def from_image(header: str) -> str:
    """The image or stage a FROM instruction builds on."""
    return next(word for word in header.split()[1:] if not word.startswith("--"))


def stage_bases(dockerfile: Path) -> dict[str, str]:
    """Map each named stage to the image or stage its FROM names."""
    return {
        header.split()[-1]: from_image(header)
        for header in stages(instructions(dockerfile))
        if header.split()[-2].upper() == "AS"
    }


def bake_contexts() -> dict[str, tuple[Path, str]]:
    """Map each named context the bake file supplies to its Dockerfile and stage."""
    bake = BAKE_PATH.read_text(encoding="utf-8")
    targets = {
        name: (
            REPO_ROOT / re.search(r'^\s*dockerfile\s*=\s*"([^"]+)"', body, re.M)[1],
            re.search(r'^\s*target\s*=\s*"([^"]+)"', body, re.M)[1],
        )
        for name, body in re.findall(
            r'^target "([\w-]+)" \{\n(.*?)^\}', bake, re.M | re.S
        )
    }
    contexts = re.findall(r'^\s*([\w-]+)\s*=\s*"target:([\w-]+)"', bake, re.M)
    return {name: targets[target] for name, target in contexts}


def base_image(dockerfile: Path, image: str) -> str:
    """Follow a FROM through earlier stages and bake contexts to an image."""
    contexts = bake_contexts()
    for _ in range(10):
        bases = stage_bases(dockerfile)
        if image in bases:
            image = bases[image]
        elif image in contexts:
            dockerfile, stage = contexts[image]
            image = stage_bases(dockerfile)[stage]
        else:
            return image
    raise AssertionError(f"{image} does not resolve to an image")


def debian_release(image: str) -> str:
    """The Debian codename an image tag names, as in debian:13-slim."""
    name, _, tag = image.split("@")[0].partition(":")
    words = re.split(r"[-.]", tag)
    if name.split("/")[-1] == "debian" and words[0] in DEBIAN_CODENAMES:
        return DEBIAN_CODENAMES[words[0]]
    named = [word for word in words if word in DEBIAN_CODENAMES.values()]
    if len(named) != 1:
        raise AssertionError(f"cannot tell which Debian release {image} is")
    return named[0]


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
    """OS package installs run again when Debian publishes security updates.

    A build argument is part of the cache key of every RUN after its ARG. Each
    stage that installs packages declares APT_REFRESH before its first install,
    and CI passes the key from .github/scripts/apt_refresh.py: the UTC date
    plus a hash of the security archive's package indexes. Debian's security
    fixes reach the next build instead of waiting behind a cached layer.
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

    def test_every_image_build_passes_the_refresh_key(self) -> None:
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
                        "${{ steps.apt-refresh.outputs.key }}",
                        step,
                    )
                    keys = [
                        earlier
                        for earlier in steps[:index]
                        if re.search(r"^\s+id: apt-refresh$", earlier, re.M)
                    ]
                    self.assertEqual(len(keys), 1, "an earlier step sets the key")
                    # An assignment fails the step if the script fails;
                    # echo "key=$(...)" would pass an empty key instead.
                    self.assertIn(
                        'key="$(python3 .github/scripts/apt_refresh.py)"\n', keys[0]
                    )
                    self.assertIn('echo "key=$key" >> "$GITHUB_OUTPUT"', keys[0])
        self.assertEqual(builds, workflow.count("uses: docker/bake-action@"))

    def test_the_key_reads_the_release_the_package_installs_build_on(self) -> None:
        # The key hashes one Debian release's security archive. Moving the
        # images to another release without the key would leave their security
        # updates waiting for the date to change again.
        for dockerfile in (BACKEND_DOCKERFILE, APPLIANCE_DOCKERFILE):
            for header, body in stages(instructions(dockerfile)).items():
                if not any(installs_packages(line) for line in body):
                    continue
                image = base_image(dockerfile, from_image(header))
                with self.subTest(dockerfile=dockerfile.parent.name, stage=header):
                    self.assertEqual(debian_release(image), apt_refresh.CODENAME)
        self.assertEqual(apt_refresh.SUITE, f"{apt_refresh.CODENAME}-security")
        self.assertIn(f"/dists/{apt_refresh.SUITE}/", apt_refresh.INDEX_URL)


if __name__ == "__main__":
    unittest.main()
