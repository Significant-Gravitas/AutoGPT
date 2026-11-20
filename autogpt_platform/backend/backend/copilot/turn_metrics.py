"""The copilot executor's count of finished turns.

It lives apart from ``executor.manager``, which keeps the executor's other
metrics, so that ``turn_tracing`` can import it at module level:
``executor.manager`` imports the processor, which imports ``turn_tracing``.
"""

from prometheus_client import Counter

# Counted once per turn, when it ends (``copilot.turn_tracing``). outcome is
# completed, cancelled or failed; engine is the one that ran the turn, or
# "none" for a turn refused before one started; origin is what started it;
# load_test is a load-test account's engine, empty for everyone else's turns.
# No label carries an id or a message.
turns_counter = Counter(
    "copilot_executor_turns_total",
    "CoPilot turns that ended, by outcome, engine, origin and load-test engine",
    ["outcome", "engine", "origin", "load_test"],
)
