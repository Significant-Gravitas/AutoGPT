"""Convert the bot's canonical CommonMark output into Telegram HTML.

Telegram's HTML parse mode is the robust dialect — MarkdownV2 requires
escaping half of ASCII. HTML-escape everything first, then re-introduce the
handful of tags Telegram supports for the CommonMark constructs we emit.
"""

import html
import re

_CODE_BLOCK_RE = re.compile(r"```(?:\w+\n)?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def to_html(text: str) -> str:
    """Render CommonMark code blocks, inline code, bold, and links as
    Telegram HTML. Everything else is escaped so user/model output can't
    inject tags."""
    escaped = html.escape(text, quote=False)
    escaped = _CODE_BLOCK_RE.sub(lambda m: f"<pre>{m.group(1).strip()}</pre>", escaped)
    escaped = _INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _LINK_RE.sub(r'<a href="\2">\1</a>', escaped)
    return escaped
