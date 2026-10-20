from pydantic import BaseModel


class CopilotSkillInfo(BaseModel):
    """User-distilled copilot skill metadata for the library UI.

    Defaults (built-in agent-building / MCP-tool guides) are intentionally
    excluded — they cannot be edited or deleted, so surfacing them in the
    user-facing list would add noise without affordances.
    """

    name: str
    description: str
    triggers: list[str] = []
    # "user" for a skill the owner saved, "marketplace" for an installed copy.
    origin: str | None = None
    # For a marketplace copy: "available" when a newer version exists but the
    # copy was edited, "merged" when a newer version was merged into the
    # owner's edits, "retired" when the listing is gone. Null when current.
    update: str | None = None


class CopilotSkillFile(BaseModel):
    """One file of a skill package, by its path relative to the skill folder
    (``scripts/run.py``) — the path the SKILL.md body references it by."""

    path: str
    size_bytes: int
    is_executable: bool = False


class CopilotSkillDetail(BaseModel):
    """Full SKILL.md content surfaced to the library expand-to-view UI."""

    name: str
    description: str
    triggers: list[str] = []
    body: str
    version: str | None = None
    is_default: bool = False
    # The package's other files (references/, scripts/, assets/).  Empty for
    # built-in defaults, which ship as one on-disk markdown file.
    files: list[CopilotSkillFile] = []


class UploadCopilotSkillRequest(BaseModel):
    """Body for the library UI's "upload skill" action.

    Carries the raw ``SKILL.md`` text (YAML frontmatter + markdown body) the
    user picked from disk; the server parses + validates it so the upload and
    the copilot's ``store_skill`` tool share one source of truth.
    """

    content: str
