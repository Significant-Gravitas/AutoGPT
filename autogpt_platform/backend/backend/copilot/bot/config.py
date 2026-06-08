"""Platform-agnostic bot config."""

# Cache TTL for AutoPilot session IDs (per channel/thread).
#
# Must be >= the thread auto-reply subscription TTL (``threads.
# THREAD_SUBSCRIPTION_TTL``, 7 days). If it's shorter, the bot keeps
# auto-replying in a thread after the session cache has expired and silently
# starts a *fresh* copilot session that has lost the whole conversation — it
# then acts on stale cross-session memory instead of the actual thread, which is
# dangerous. Each turn refreshes this TTL, so an active conversation never loses
# context within the subscription window.
SESSION_TTL = 7 * 86400  # 7 days — matches the thread subscription window
