"""Convert outbound Markdown text to Slack mrkdwn + resolve user mentions."""

import re


def to_mrkdwn(text: str, mentionable_users: tuple[tuple[str, str], ...] = ()) -> str:
    """Render the handler's Markdown output in Slack's mrkdwn dialect."""
    text = _convert_bold(text)
    text = _convert_links(text)
    text = _resolve_mentions(text, mentionable_users)
    return text


def _convert_bold(text: str) -> str:
    # Markdown **bold** → mrkdwn *bold*.
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)


def _convert_links(text: str) -> str:
    # Markdown [label](url) → mrkdwn <url|label>.
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)


def _resolve_mentions(text: str, mentionable_users: tuple[tuple[str, str], ...]) -> str:
    """Substitute `@DisplayName` with `<@U123>` for users on the allowlist.

    Same allowlist defence as the Discord adapter: anyone not on the list
    stays as plain text, even if the LLM hallucinates a mention.
    """
    if not mentionable_users:
        return text
    rendered = text
    # Longest names first so "@John Smith" matches before "@John".
    for display_name, user_id in sorted(
        mentionable_users, key=lambda pair: -len(pair[0])
    ):
        # Word-bounded so "@Name" inside emails/URLs is left alone.
        pattern = re.compile(
            rf"(?<![\w@]){re.escape(f'@{display_name}')}(?!\w)",
            re.IGNORECASE,
        )
        # Callable replacement avoids backref interpretation of user_id.
        rendered = pattern.sub(lambda _m, uid=user_id: f"<@{uid}>", rendered)
    return rendered
