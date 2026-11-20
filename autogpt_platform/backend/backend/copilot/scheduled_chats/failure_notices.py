"""What a failing scheduled chat tells its owner (SECRT-2879).

Either it was paused, after three runs in a row failed outright the same way,
or it is still running but the same step has failed three times in a row.
Either way the note says what broke and offers the fix: reconnect the
provider, link the chat platform, resume. One that could not be paused says
the same without that claim. Like every note here, it is for the owner and
hidden from the model.
"""

from backend.copilot.scheduled_chats.notices import (
    BILLING_PATH,
    INTEGRATIONS_PATH,
    NoticeAction,
    NoticeLink,
    NoticeReconnect,
    NoticeResume,
    ScheduledChatNotice,
)
from backend.copilot.scheduled_chats.outcomes import RunFailure

BOTS_PATH = "/settings/bots"

# Where ``str.title()`` gets a provider's name wrong.
_PROVIDER_NAMES = {
    "github": "GitHub",
    "gitlab": "GitLab",
    "hubspot": "HubSpot",
    "linkedin": "LinkedIn",
    "openai": "OpenAI",
    "youtube": "YouTube",
}


def provider_name(provider: str | None) -> str:
    if not provider:
        return "a connected service"
    return _PROVIDER_NAMES.get(provider, provider.replace("_", " ").title())


def paused_notice(
    failure: RunFailure,
    *,
    title: str,
    failures: int,
    streak_id: str,
    paused: bool = True,
) -> ScheduledChatNotice:
    """Three runs in a row failed outright the same way. *paused* is False
    when the schedule could not be paused, so it is still firing."""
    name = provider_name(failure.provider)
    runs = f"Its last {failures} runs"
    if failure.kind == "connection" and failure.provider:
        headline = f"“{title}” can't use {name}"
        cause = (
            f"{runs} failed because the {name} connection is missing or was " "refused."
        )
        fix = "Reconnect it and the chat starts again." if paused else "Reconnect it."
    elif failure.kind == "connection":
        # Its tool didn't say which account: the owner finds it in Settings,
        # and reconnecting there resumes nothing by itself.
        headline = f"“{title}” can't use one of its connections"
        cause = (
            f"{runs} failed because a connection it needs is missing or was refused."
        )
        fix = (
            "Reconnect it in Settings, then resume."
            if paused
            else "Reconnect it in Settings."
        )
    elif failure.kind == "chat_link":
        headline = f"“{title}” can't post to {_platform(failure)}"
        cause = (
            f"{runs} couldn't post, because {_no_link(failure)} is linked to "
            "your account."
        )
        fix = "Link one, then resume." if paused else "Link one."
    elif failure.kind == "budget":
        limit = "step" if failure.key == "turn:max_turns_exhausted" else "spending"
        headline = f"“{title}” keeps hitting its {limit} limit"
        cause = (
            f"{runs} stopped at the {limit} limit one run may use, before finishing."
        )
        fix = (
            "Resume it to try again, or give it a smaller job."
            if paused
            else "Give it a smaller job."
        )
    else:
        headline = f"“{title}” keeps failing"
        cause = f"{runs} failed the same way."
        fix = (
            "Open the chat to see the error, then resume."
            if paused
            else "Open the chat to see the error."
        )
    if not paused:
        return ScheduledChatNotice(
            reason=failure.kind,
            headline=headline,
            body=f"{cause} {fix}",
            actions=_fixes(failure),
            window_key=f"failing:{streak_id}",
        )
    return ScheduledChatNotice(
        reason=failure.kind,
        headline=f"Paused: {headline}",
        body=f"{cause} {fix}",
        actions=[*_fixes(failure), NoticeResume()],
        window_key=f"paused:{streak_id}",
    )


def failing_notice(
    failure: RunFailure, *, title: str, failures: int, streak_id: str
) -> ScheduledChatNotice:
    """The schedule keeps running: the rest of what it does still works."""
    name = provider_name(failure.provider)
    runs = f"The last {failures} runs"
    if failure.kind == "connection" and failure.provider:
        headline = f"“{title}” can't use {name}"
        body = (
            f"{runs} went ahead without {name}, because the connection is "
            "missing or was refused. Reconnect it to bring it back."
        )
    elif failure.kind == "connection":
        headline = f"“{title}” can't use one of its connections"
        body = (
            f"{runs} went ahead without a connection they need, which is "
            "missing or was refused. Reconnect it in Settings to bring it back."
        )
    elif failure.kind == "chat_link":
        headline = f"“{title}” can't post to {_platform(failure)}"
        body = (
            f"{runs} went ahead without posting, because {_no_link(failure)} "
            "is linked to your account."
        )
    else:
        headline = f"“{title}” keeps failing at one step"
        body = f"{runs} went ahead without it. Open the chat to see the error."
    return ScheduledChatNotice(
        reason=failure.kind,
        headline=headline,
        body=body,
        actions=_fixes(failure),
        window_key=f"failing:{streak_id}",
    )


def pause_summary(failure: RunFailure) -> str:
    """Why a scheduled chat is paused, in a few words, for the lists."""
    if failure.kind == "connection" and failure.provider:
        return f"Paused: can't use {provider_name(failure.provider)}"
    if failure.kind == "connection":
        return "Paused: can't use one of its connections"
    if failure.kind == "chat_link":
        return f"Paused: can't post to {_platform(failure)}"
    if failure.key == "turn:max_turns_exhausted":
        return "Paused: keeps hitting its step limit"
    if failure.kind == "budget":
        return "Paused: keeps hitting its spending limit"
    return "Paused: keeps failing"


def resumed_notice(*, title: str, streak_id: str) -> ScheduledChatNotice:
    return ScheduledChatNotice(
        reason="resumed",
        headline=f"Resumed “{title}”",
        body="It runs again at its next scheduled time.",
        actions=[],
        window_key=f"resumed:{streak_id}",
    )


def _fixes(failure: RunFailure) -> list[NoticeAction]:
    if failure.kind == "connection" and failure.provider:
        label = f"Reconnect {provider_name(failure.provider)}"
        return [NoticeReconnect(label=label, provider=failure.provider)]
    if failure.kind == "connection":
        return [NoticeLink(label="Open settings", href=INTEGRATIONS_PATH)]
    if failure.kind == "chat_link":
        return [NoticeLink(label="Open bot settings", href=BOTS_PATH)]
    if failure.key == "turn:max_budget_exhausted":
        return [NoticeLink(label="Upgrade", href=BILLING_PATH)]
    return []


def _platform(failure: RunFailure) -> str:
    return provider_name(failure.provider) if failure.provider else "a chat platform"


def _no_link(failure: RunFailure) -> str:
    if failure.provider:
        return f"no {provider_name(failure.provider)} server or chat"
    return "no server or chat"
