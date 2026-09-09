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


class CopilotSkillDetail(BaseModel):
    """Full SKILL.md content surfaced to the library expand-to-view UI."""

    name: str
    description: str
    triggers: list[str] = []
    body: str
    version: str | None = None
    is_default: bool = False
    # Sibling files in the same skill folder (references/, scripts/,
    # assets/, etc.) — the workspace paths the model can reach via
    # ``read_workspace_file``.  Empty for built-in defaults since they
    # ship as on-disk markdown and have no sibling artefacts.
    sibling_files: list[str] = []


class UploadCopilotSkillRequest(BaseModel):
    """Body for the library UI's "upload skill" action.

    Carries the raw ``SKILL.md`` text (YAML frontmatter + markdown body) the
    user picked from disk; the server parses + validates it so the upload and
    the copilot's ``store_skill`` tool share one source of truth.
    """

    content: str
