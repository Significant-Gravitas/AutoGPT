"""Per-turn options typed in front of a chat message.

``expert=Bolt mode=smart fix the build`` routes the turn to the expert named
Bolt on the Advanced tier and leaves ``fix the build`` as the message. Only
leading tokens are read, so ``mode=`` in the middle of a sentence stays text.
"""

import re

from pydantic import BaseModel

from backend.copilot.config import CopilotLLMModel

MODE_TIERS: dict[str, CopilotLLMModel] = {
    "fast": "standard",
    "standard": "standard",
    "smart": "advanced",
    "advanced": "advanced",
}

_LEADING_OPTION = re.compile(
    r"""\A\s*(?P<key>expert|mode)=(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>\S+))""",
    re.IGNORECASE,
)


class TurnOptions(BaseModel):
    """What a message asked for. ``expert_name`` is resolved to
    ``expert_id`` against the owner's hires before the turn runs; a slash
    command fills ``expert_id`` directly."""

    expert_name: str | None = None
    expert_id: str | None = None
    model: CopilotLLMModel | None = None

    def merged(self, newer: "TurnOptions") -> "TurnOptions":
        """``newer``'s choices win; what it leaves unset keeps ours."""
        return self.model_copy(update=newer.model_dump(exclude_none=True))


def parse_turn_options(text: str) -> tuple[TurnOptions, str]:
    """Strip leading ``expert=`` / ``mode=`` tokens off ``text``.

    Keys are case-insensitive and names may be quoted
    (``expert="Bolt the Builder"``). An unknown key, or a ``mode`` value that
    isn't a tier, ends the options and stays in the text.
    """
    options = TurnOptions()
    rest = text
    while match := _LEADING_OPTION.match(rest):
        value = _option_value(match)
        if match.group("key").lower() == "mode":
            tier = MODE_TIERS.get(value.lower())
            if tier is None:
                break
            options.model = tier
        elif value.strip():
            options.expert_name = value.strip()
        else:
            break
        rest = rest[match.end() :]
    if rest is text:
        return options, text
    return options, rest.lstrip()


def _option_value(match: re.Match[str]) -> str:
    for group in ("dq", "sq", "bare"):
        value = match.group(group)
        if value is not None:
            return value
    return ""
