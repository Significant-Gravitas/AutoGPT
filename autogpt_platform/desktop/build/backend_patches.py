"""Changes to the backend, made to the bundle's copy of it.

Native compatibility changes that remain outside the shared platform source
are applied to the copy the build puts in the bundle, before it is compiled.
The build never edits the source checkout.

Each patch replaces one exact piece of text. The build stops when that text
is not in the file exactly once: upstream changed it, and somebody has to
look. runtime/tests/test_backend_patches.py finds that out earlier, against
the backend in the repository, on every run of the unit tests.

Each patch is also offered upstream, as upstream/<name>.patch (the same
change as a diff, with a test where one makes sense). When upstream has
merged one, delete it here and there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Patch:
    # Also the name of the diff offered upstream: upstream/<name>.patch.
    name: str
    # Inside the backend directory (the one that holds schema.prisma).
    file: str
    original: str
    replacement: str
    why: str

    def apply(self, text: str) -> str:
        found = text.count(self.original)
        if found != 1:
            raise ValueError(
                f"the backend patch {self.name!r} no longer fits backend/{self.file}: the text "
                f"it replaces is there {found} times, not once.\n\n{self.original}\n"
                "Upstream changed it. If upstream now does what the patch did, delete the "
                f"patch from desktop/build/backend_patches.py and desktop/upstream/{self.name}"
                ".patch; otherwise write both again for the new code "
                "(desktop/upstream/README.md)."
            )
        return text.replace(self.original, self.replacement)

    def is_applied(self, text: str) -> bool:
        return text.count(self.replacement) == 1 and self.original not in text


PATCHES = (
    Patch(
        name="copilot-workspace-prefix",
        file="backend/copilot/tools/sandbox.py",
        original='WORKSPACE_PREFIX = "/tmp/copilot-"\n',
        replacement=(
            "# COPILOT_WORKSPACE_PREFIX moves the workspaces on a host with no /tmp (native\n"
            "# Windows). normpath is the identity on POSIX; on Windows it gives the prefix\n"
            "# the separators that the normalised paths compared with it have.\n"
            "WORKSPACE_PREFIX = os.path.normpath(\n"
            '    os.environ.get("COPILOT_WORKSPACE_PREFIX") or "/tmp/copilot-"\n'
            ")\n"
        ),
        why=(
            "No AutoPilot turn can start on Windows. make_session_path, _make_sdk_cwd and "
            "_cleanup_sdk_tool_results compare os.path.normpath(path) with this prefix; "
            "normpath turns / into \\ there, so the comparison never holds and the turn "
            "ends with 'Session path escaped prefix'. With the prefix normalised too, it "
            "holds, and on POSIX nothing changes. The variable is there because a "
            "drive-relative \\tmp is the top of whatever drive the backend runs from: "
            "shared by every user of the machine. The desktop sets it, on Windows only, "
            "to a directory inside its data directory "
            "(runtime/autogpt_desktop/settings.py)."
        ),
    ),
    Patch(
        name="claude-subscription-log",
        file="backend/copilot/sdk/subscription.py",
        original=(
            '            "Claude subscription auth: method=%s, email=%s",\n'
            '            status.get("authMethod"),\n'
            '            status.get("email"),\n'
        ),
        replacement=(
            '            "Claude subscription auth: method=%s",\n'
            '            status.get("authMethod"),\n'
        ),
        why=(
            "With AutoPilot on the user's own Claude Code sign-in, this line writes the "
            "email address of their Claude account into the service log, which people "
            "attach to bug reports. Nothing reads it back."
        ),
    ),
)


def apply(backend: Path) -> None:
    """Patch the backend copied to `backend` (<bundle>/backend)."""
    for patch in PATCHES:
        path = backend / patch.file
        path.write_text(patch.apply(read(path)), encoding="utf-8", newline="\n")
        print(f"  patched backend/{patch.file}: {patch.name}")


def check(backend: Path) -> None:
    """Refuse a bundle whose backend is not the patched one."""
    missing = [
        patch.name
        for patch in PATCHES
        if not patch.is_applied(read(backend / patch.file))
    ]
    if missing:
        raise RuntimeError(
            f"the bundled backend lacks the build-time patches {missing} "
            "(build/backend_patches.py); run the backend step"
        )


def read(path: Path) -> str:
    """With \\n line ends, whatever the checkout has: a Windows checkout's
    are \\r\\n, and the texts above are written with \\n."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")
