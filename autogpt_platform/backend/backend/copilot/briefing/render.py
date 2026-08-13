"""Markdown rendering for the briefing posted into the copilot thread."""

import re

from .models import BriefingContent

# Agent names, AI-generated summaries and review instructions can all
# originate from a third-party/marketplace agent. Escaping the markdown
# metacharacters that carry structure stops that text from breaking out of
# the link syntax it is interpolated into and spoofing the label or target
# rendered in the user's thread.
_MARKDOWN_META_RE = re.compile(r"([\\`*_\[\]()<>])")


def render_briefing_markdown(content: BriefingContent) -> str:
    lines = ["## ☀️ Your morning briefing", ""]
    if content.narrative:
        # Escaped like every other interpolated string: the narrative is model
        # output derived from agent-supplied text, so it is no more trusted
        # than the outcome titles it was written from.
        lines.extend([_md(content.narrative), ""])
    if content.run_items:
        lines.append("**What ran**")
        for item in content.run_items:
            who = f"{_md(item.expert_name)}: " if item.expert_name else ""
            outcome = "completed" if item.status == "COMPLETED" else "failed"
            name = _md_link(_md(item.agent_name), item.link)
            lines.append(f"- {who}{name} — {outcome}")
        lines.append("")
    found = [item for item in content.run_items if item.summary]
    if found:
        lines.append("**What was found**")
        lines.extend(
            f"- **{_md(item.agent_name)}**: {_md(item.summary or '')}" for item in found
        )
        lines.append("")
    if content.decision_items:
        total = max(content.decision_total, len(content.decision_items))
        lines.append(f"**Needs your decision ({total})**")
        lines.extend(
            f"- {_md_link(_md(d.title), d.link)}" for d in content.decision_items
        )
        remaining = total - len(content.decision_items)
        if remaining > 0:
            lines.append(f"- …and {remaining} more on your home page")
    return "\n".join(lines).strip()


def _md(text: str) -> str:
    """Escape untrusted text for inline interpolation into markdown."""
    collapsed = " ".join(text.split())
    return _MARKDOWN_META_RE.sub(r"\\\1", collapsed)


def _md_link(label: str, target: str | None) -> str:
    """Render ``label`` as a markdown link, or as plain text if it can't be.

    Composed targets are relative, percent-encoded paths. Enforcing that in
    code — rather than by convention — keeps an absolute or ``javascript:``
    target from ever reaching the user's thread as a clickable link.
    """
    if not target or not target.startswith("/") or target.startswith("//"):
        return label
    return f"[{label}]({target})"
