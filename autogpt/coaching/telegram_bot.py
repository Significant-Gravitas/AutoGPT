"""Telegram bot front-end for the ABN Consulting AI Co-Navigator.

Each Telegram user gets their own coaching session.  The bot talks directly
to the coaching session layer (no HTTP round-trip needed).

Usage
-----
Set these environment variables (same .env as the API):

    TELEGRAM_BOT_TOKEN=<your BotFather token>
    ANTHROPIC_API_KEY=...
    SUPABASE_URL=...
    SUPABASE_SERVICE_KEY=...
    COACHING_API_KEY=...   (unused by the bot but loaded by coaching_config)

Run:

    python -m autogpt.coaching.telegram_bot

Commands
--------
/start   – open a new coaching session (or resume greeting if one is active)
/end     – close the current session and receive the structured summary
/cancel  – discard the current session without saving
/help    – show this help message
Any other text is forwarded to the active coaching session.
"""
from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Mapping from Telegram user_id → active CoachingSession
_sessions: dict[int, object] = {}  # value type: CoachingSession


def _get_coaching_session(tg_user_id: int):
    return _sessions.get(tg_user_id)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Open a new coaching session for this Telegram user."""
    from autogpt.coaching.session import CoachingSession
    from autogpt.coaching.storage import get_past_sessions, get_user_objectives

    tg_user = update.effective_user
    tg_user_id = tg_user.id
    client_name = tg_user.full_name or f"User {tg_user_id}"
    client_id = f"tg_{tg_user_id}"

    if tg_user_id in _sessions:
        await update.message.reply_text(
            "You already have an active session. Send /end to close it or keep chatting!"
        )
        return

    # Try to load past context from Supabase (best-effort — won't fail if offline)
    objectives = []
    past_sessions = []
    try:
        objectives = get_user_objectives(client_id)
        past_sessions = get_past_sessions(client_id, limit=3)
    except Exception as exc:
        logger.warning("Could not load user context for tg_user=%d: %s", tg_user_id, exc)

    session = CoachingSession(
        client_id=client_id,
        client_name=client_name,
        objectives=objectives,
        past_sessions=past_sessions,
    )
    _sessions[tg_user_id] = session

    await update.message.reply_text("Starting your coaching session... one moment.")

    try:
        opening = await asyncio.get_event_loop().run_in_executor(None, session.open)
    except Exception as exc:
        logger.error("session.open failed for tg_user=%d: %s", tg_user_id, exc)
        del _sessions[tg_user_id]
        await update.message.reply_text(
            "Sorry, I couldn't start the session right now. Please try again later."
        )
        return

    await update.message.reply_text(opening)


async def cmd_end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """End the active session and send the structured summary."""
    from autogpt.coaching.storage import save_session

    tg_user_id = update.effective_user.id
    session = _get_coaching_session(tg_user_id)
    if session is None:
        await update.message.reply_text(
            "No active session. Use /start to begin a new coaching session."
        )
        return

    await update.message.reply_text(
        "Wrapping up your session — extracting summary... please wait."
    )

    try:
        summary = await asyncio.get_event_loop().run_in_executor(
            None, session.extract_summary
        )
        await asyncio.get_event_loop().run_in_executor(None, save_session, summary)
    except Exception as exc:
        logger.error("Session end failed for tg_user=%d: %s", tg_user_id, exc)
        await update.message.reply_text(
            "Something went wrong saving your session. Your data may not have been saved."
        )
        return
    finally:
        _sessions.pop(tg_user_id, None)

    # Format a readable summary
    wl = summary.weekly_log
    lines = [
        f"*Session Summary*",
        f"",
        f"*Focus goal:* {wl.focus_goal or '—'}",
        f"*Mood:* {wl.mood_indicator or '—'}",
        f"*Environmental changes:* {wl.environmental_changes or '—'}",
    ]

    if wl.key_results:
        lines.append("\n*Key Results:*")
        for kr in wl.key_results:
            lines.append(f"  • {kr.description} — {kr.status_pct}%")

    if wl.obstacles:
        unresolved = [o.description for o in wl.obstacles if not o.resolved]
        if unresolved:
            lines.append("\n*Open obstacles:*")
            for obs in unresolved:
                lines.append(f"  ⚠ {obs}")

    lines += [
        f"",
        f"*Alert level:* {summary.alerts.level.value.upper()}",
        f"_{summary.alerts.reason}_",
        f"",
        f"*Coach note:* {summary.summary_for_coach}",
        f"",
        f"Session saved. See you next week! 🚢",
    ]

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Discard the current session without saving."""
    tg_user_id = update.effective_user.id
    if _sessions.pop(tg_user_id, None) is None:
        await update.message.reply_text("No active session to cancel.")
    else:
        await update.message.reply_text(
            "Session discarded. Nothing was saved. Use /start to begin again."
        )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "*ABN Co-Navigator — Telegram Commands*\n\n"
        "/start  — Begin a new coaching session\n"
        "/end    — Close your session and receive a summary\n"
        "/cancel — Discard the current session (no save)\n"
        "/help   — Show this message\n\n"
        "During a session just type naturally — the Navigator will guide you through "
        "your weekly OKR check-in."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forward user text to the active coaching session."""
    tg_user_id = update.effective_user.id
    session = _get_coaching_session(tg_user_id)

    if session is None:
        await update.message.reply_text(
            "No active session. Use /start to begin your coaching session."
        )
        return

    user_text = update.message.text or ""
    if not user_text.strip():
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        reply = await asyncio.get_event_loop().run_in_executor(
            None, session.chat, user_text
        )
    except Exception as exc:
        logger.error("session.chat failed for tg_user=%d: %s", tg_user_id, exc)
        await update.message.reply_text(
            "I'm having trouble responding right now. Please try again in a moment."
        )
        return

    await update.message.reply_text(reply)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("end", cmd_end))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("ABN Co-Navigator Telegram bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
