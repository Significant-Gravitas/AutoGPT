"""Assembly of the SDK turn's system prompt.

The turn's first launch and the building-mode restart build their prompt
here, so the restart's prompt is the launch's prompt plus the guide.
"""

from pydantic import BaseModel, ConfigDict

from backend.copilot.prompting import get_sdk_supplement


class SystemPromptParts(BaseModel):
    """The pieces of the system prompt settled at the start of a turn.

    The building guide is passed to :func:`assemble_system_prompt` instead,
    because the building-mode restart adds it mid-turn.
    """

    model_config = ConfigDict(frozen=True)

    base: str
    use_e2b: bool
    expert_session: bool
    delegation: str
    oversight: str
    team_building: str
    chat_platform: str
    graphiti: str
    auto_mode: str
    expert_identity: str


def assemble_system_prompt(parts: SystemPromptParts, *, building_suffix: str) -> str:
    """Join the turn's prompt pieces in their one order.

    The building guide and the expert identity go last so the shared prefix
    stays byte-identical across sessions that have neither.
    """
    return (
        parts.base
        + get_sdk_supplement(use_e2b=parts.use_e2b, expert_session=parts.expert_session)
        + parts.delegation
        + parts.oversight
        + parts.team_building
        + parts.chat_platform
        + parts.graphiti
        + parts.auto_mode
        + building_suffix
        + parts.expert_identity
    )
