import pytest

from .workspace_internal import is_internal_workspace_file


@pytest.mark.parametrize(
    "name, path, metadata",
    [
        # Large tool outputs parked for Otto to read back.
        (
            "sdk-926fe642657d.json",
            "/sessions/s1/tool-outputs/sdk-926fe642657d.json",
            {"purpose": "tool-output"},
        ),
        # The same, written before the purpose was recorded.
        ("toolu_01ABC.json", "/sessions/s1/tool-results/toolu_01ABC.json", {}),
        # A writer that only makes working files says so.
        ("agent.json", "/sessions/s1/agent.json", {"purpose": "working-file"}),
        # The scratchpad the prompt asks for; the model names it itself.
        (
            "build_state.json",
            "/sessions/s1/build_state.json",
            {"origin": "agent-created"},
        ),
        ("Build_State.json", "/build_state.json", {}),
        ("._browser_state.json", "/sessions/s1/._browser_state.json", {}),
    ],
)
def test_otto_working_files_are_internal(name: str, path: str, metadata: dict) -> None:
    assert is_internal_workspace_file(name=name, path=path, metadata=metadata)


@pytest.mark.parametrize(
    "name, path, metadata",
    [
        # What the user asked for: images, exports, documents.
        ("sticker-01.png", "/sessions/s1/sticker-01.png", {"origin": "agent-created"}),
        ("sticker-designs.tar.gz", "/sessions/s1/sticker-designs.tar.gz", {}),
        ("report.pdf", "/report.pdf", {}),
        # Anything the user uploads is theirs, whatever it is called.
        ("build_state.json", "/build_state.json", {"origin": "user-upload"}),
        (
            "x.json",
            "/sessions/s1/tool-outputs/x.json",
            {"origin": "user-upload", "purpose": "tool-output"},
        ),
        # A stated purpose wins over where a file sits.
        (
            "report.json",
            "/sessions/s1/tool-outputs/report.json",
            {"purpose": "deliverable"},
        ),
        # Only the reserved directory under a session root is Otto's.
        ("data.json", "/my-pipeline/tool-outputs/data.json", {}),
        ("data.json", "/sessions/s1/nested/tool-outputs/data.json", {}),
        # A state file the model named itself is not recognised, so it shows.
        ("watcher.json", "/sessions/s1/watcher.json", {"origin": "agent-created"}),
    ],
)
def test_everything_else_is_a_deliverable(name: str, path: str, metadata: dict) -> None:
    assert not is_internal_workspace_file(name=name, path=path, metadata=metadata)


def test_no_metadata_reads_as_none() -> None:
    assert not is_internal_workspace_file(name="a.png", path="/a.png", metadata=None)


@pytest.mark.parametrize(
    "name, path, metadata",
    [
        # A hired expert's skill package, installed from the marketplace.
        (
            "SKILL.md",
            "/experts/e1/skills/copywriting/SKILL.md",
            {"kind": "copilot_skill", "skill_origin": "marketplace"},
        ),
        ("LICENSE", "/experts/e1/skills/copywriting/LICENSE", {}),
        ("sendgrid.md", "/experts/e1/skills/email/references/sendgrid.md", {}),
        # The account's own skills, which personal Otto uses.
        ("SKILL.md", "/skills/deep-research/SKILL.md", {"skill_origin": "user"}),
        ("run.py", "/skills/deep-research/scripts/run.py", {"executable": True}),
    ],
)
def test_skill_packages_are_internal(name: str, path: str, metadata: dict) -> None:
    assert is_internal_workspace_file(name=name, path=path, metadata=metadata)


@pytest.mark.parametrize(
    "name, path",
    [
        # Only the skill folders are; a file merely named like one is not.
        ("SKILL.md", "/SKILL.md"),
        ("SKILL.md", "/my-skills/copywriting/SKILL.md"),
        ("skills.md", "/sessions/s1/skills/skills.md"),
        ("plan.md", "/experts/e1/notes/skills/plan.md"),
    ],
)
def test_files_outside_skill_folders_are_not(name: str, path: str) -> None:
    assert not is_internal_workspace_file(name=name, path=path, metadata={})
