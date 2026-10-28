"""Dream pass — scheduled offline memory consolidation.

Three-phase pipeline (consolidate → recombine → sanitize) that walks a
user's recent episodes + active facts, proposes new tentative findings,
demotes stale or contradicted ones, and writes a summary chat session.

Designed to run on a per-user cron in user-local time (default 03:00),
with an admin-triggered on-demand path for testing.
"""
