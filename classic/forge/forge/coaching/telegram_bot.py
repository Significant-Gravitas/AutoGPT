"""Telegram bot interface for the ABN Co-Navigator coaching sessions."""
from __future__ import annotations

import asyncio
import logging
from typing import Dict

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from autogpt.coaching.config import coaching_config
from autogpt.coaching.session import CoachingSession

logger = logging.getLogger(__name__)

# Conversation states
WAITING_NAME, CHATTING = range(2)

# Active sessions: telegram user_id → CoachingSession
_sessions: Dict[int, CoachingSession] = {}


# ── Command handlers ──────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id in _sessions:
        await update.message.reply_text(
            "You already have an active session. Keep chatting, "
            "or use /done to wrap up and receive your summary."
        )
        return CHATTING
    await update.message.reply_text(
        "👋 *Welcome to the ABN Consulting AI Co-Navigator!*\n\n"
        "I'll guide you through a structured coaching session — reviewing "
        "your OKRs, logging progress, and surfacing obstacles.\n\n"
        "What's your name?",
        parse_mode="Markdown",
    )
    return WAITING_NAME


async def receive_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    name = update.message.text.strip()
    if not name or len(name) > 100:
        await update.message.reply_text("Please enter a valid name (up to 100 characters).")
        return WAITING_NAME

    await update.message.reply_text("Starting your session… ⏳")
    try:
        session = CoachingSession(client_id=f"telegram_{user_id}", client_name=name)
        _sessions[user_id] = session
        opening = session.open()
        await update.message.reply_text(opening)
        await update.message.reply_text(
            "_Tip: send /done when you're ready to end the session and get your summary._",
            parse_mode="Markdown",
        )
        return CHATTING
    except Exception:
        logger.exception("Failed to start session for telegram user %s", user_id)
        await update.message.reply_text(
            "Sorry, I couldn't start your session. Please try again with /start."
        )
        return ConversationHandler.END


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    session = _sessions.get(user_id)
    if not session:
        await update.message.reply_text("No active session — use /start to begin.")
        return ConversationHandler.END

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        reply = session.chat(update.message.text)
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Chat error for telegram user %s", user_id)
        await update.message.reply_text("Sorry, something went wrong. Please try again.")
    return CHATTING


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    session = _sessions.get(user_id)
    if not session:
        await update.message.reply_text("No active session to end.")
        return ConversationHandler.END

    await update.message.reply_text("Wrapping up your session… ⏳")
    try:
        from autogpt.coaching.storage import save_session
        summary = session.extract_summary()
        save_session(summary)
        del _sessions[user_id]
        await update.message.reply_text(_format_summary(summary), parse_mode="Markdown")
    except Exception:
        logger.exception("End session error for telegram user %s", user_id)
        _sessions.pop(user_id, None)
        await update.message.reply_text(
            "Sorry, I couldn't generate your summary. Your session has been cleared."
        )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _sessions.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "Session cancelled. Use /start whenever you're ready to begin again."
    )
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*ABN Co-Navigator — Commands*\n\n"
        "/start — Begin a new coaching session\n"
        "/done — End session and receive your summary\n"
        "/cancel — Cancel without saving\n"
        "/help — Show this message",
        parse_mode="Markdown",
    )


# ── Summary formatter ─────────────────────────────────────────────────────────

def _format_summary(summary) -> str:
    lines = [f"✅ *Session Summary — {summary.client_name}*\n"]
    log = summary.weekly_log
    if log:
        if log.focus_goal:
            lines.append(f"🎯 *Focus:* {log.focus_goal}")
        if log.key_results:
            lines.append("\n📊 *Key Results:*")
            for kr in log.key_results:
                dot = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(kr.status_color, "⚪")
                lines.append(f"  {dot} {kr.description}: {kr.status_pct}%")
        unresolved = [o for o in (log.obstacles or []) if not o.resolved]
        if unresolved:
            lines.append("\n⚠️ *Open Obstacles:*")
            for o in unresolved:
                lines.append(f"  • {o.description}")
        if log.mood_indicator:
            mood = ["😔", "😐", "🙂", "😊", "🌟"][min(log.mood_indicator - 1, 4)]
            lines.append(f"\n{mood} *Mood:* {log.mood_indicator}/5")
    if summary.alerts:
        lines.append("\n🚨 *Alerts:*")
        for alert in summary.alerts:
            lines.append(f"  • {alert.reason}")
    if summary.summary_for_coach:
        excerpt = summary.summary_for_coach[:280]
        lines.append(f"\n📝 *Coach Notes:* {excerpt}…")
    lines.append(f"\n📅 [Book your next session]({coaching_config.coach_calendly_url})")
    return "\n".join(lines)


# ── Bot runner ────────────────────────────────────────────────────────────────

def _build_app(token: str) -> Application:
    app = Application.builder().token(token).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_name)],
            CHATTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
                CommandHandler("done", done),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start", start),
        ],
    )
    app.add_handler(conv)
    app.add_handler(CommandHandler("help", help_command))
    return app


async def run_polling(token: str) -> None:
    """Start the bot in polling mode (suitable for Railway)."""
    application = _build_app(token)
    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot polling started")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Telegram bot stopped")
