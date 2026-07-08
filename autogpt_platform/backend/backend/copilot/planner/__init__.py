"""Two-phase planner/executor split for the baseline copilot path.

An expensive ``planner`` model produces a persisted structured plan first,
then a cheaper ``executor`` model runs the normal tool-call loop consuming
that plan, with a bounded re-plan escalation on failure. The whole behaviour
is gated behind the ``copilot-planner-executor`` LaunchDarkly flag (default
OFF) — see ``copilot.model_router.is_planner_executor_enabled``.
"""
