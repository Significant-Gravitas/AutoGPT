"""Framing shared by the two cross-expert handoff tools.

``delegate_to_expert`` and ``handoff_to_expert`` both open their prompt with a
``[...]`` preamble that interpolates the calling expert's name. Those names are
user-authored, so the sanitiser below is a safety invariant, not formatting —
and it lived in both files until a hardening landed in one and not the other,
leaving the twin forgeable. It lives here now so there is one place to fix.
"""

# Long enough for a real name, short enough that a crafted one cannot bury the
# preamble's own instructions under padding.
CALLER_NAME_LIMIT = 80

# The preamble delimits itself with square brackets, so a name containing them
# can close the framing early and open a block of its own.
_FRAMING_DELIMITERS = str.maketrans("", "", "[]")


def safe_caller_name(caller: str) -> str:
    """Collapse *caller* to a single bracket-free line, capped and non-empty.

    Truncating after stripping matters: a name that is all brackets must not
    spend the budget and then collapse to nothing.
    """
    one_line = " ".join(caller.split()).translate(_FRAMING_DELIMITERS)
    return one_line[:CALLER_NAME_LIMIT].strip() or "a teammate"
