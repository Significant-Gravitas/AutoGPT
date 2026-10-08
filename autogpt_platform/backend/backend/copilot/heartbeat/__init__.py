"""The copilot heartbeat: a periodic, unattended check on the user's behalf.

Modelled on OpenClaw's heartbeat and Grok Bot's proactive wake. A user who
switches it on writes a short checklist (like a ``HEARTBEAT.md``); every
``interval_minutes`` inside their active hours the scheduler runs one isolated
copilot turn on the cheap model tier with that checklist. The turn alerts only
through ``heartbeat_respond`` or a substantial reply; ``NO_REPLY`` and short
acknowledgements are dropped, and the same alert is not sent twice in a day.

- ``config``: the per-user settings (stored in ``User.metadata``), the
  active-hours window and the empty-checklist test.
- ``state``: the Redis markers (last run, last alert, the turn's explicit
  answer).
- ``prompt`` and ``suppress``: what the turn is told, and what of its answer
  reaches the user.
- ``delivery``: the user's main thread, the WebSocket/push notification, and
  linked chat platforms.
- ``runner``: one run, end to end; ``scheduling``: the APScheduler job.
"""
