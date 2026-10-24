"""Recognise the preamble a teammate task opens with, so display paths can drop it.

``delegate_to_expert`` and ``handoff_to_expert`` store the task in the
receiving thread as ``[<preamble>]\n\n<task>``. The preamble is server framing
for the model and names the user ("a teammate on Priya's team"), so a public
share must not carry it. The owner's UI already hides it with the matching
regex in ``userMessageParts.ts``.

The builders live in the tool modules, which this module can't import (the
share sanitizer that needs it sits below them), so the wording is repeated
here. ``delegation_preamble_cases.json`` ties all three together: a rewording
in a builder fails ``delegation_preamble_cases_test.py`` until this matcher
and the frontend's follow. The team owner matches any bracket-free name, which
includes "this user's team", the wording rows were stored with before the
name was added.
"""

import re

# Bounded, so a crafted line of repeated ", a teammate on" can't send the
# match into quadratic backtracking. 80 covers every name a preamble holds:
# callers are capped at CALLER_NAME_LIMIT (80), users at USER_NAME_LIMIT (40),
# and old rows say "this user".
NAME_MAX_LENGTH = 80
_NAME = rf"[^\[\]\r\n]{{1,{NAME_MAX_LENGTH}}}"

_PREAMBLES = (
    re.compile(
        rf"^\[Delegated task from {_NAME}, a teammate on {_NAME}'s team — not "
        r"the user\. They cannot see your thread, so report the outcome in your "
        r"final message\. If the task needs something only the user can "
        r"provide, say what is missing instead of guessing\.\]\r?\n\r?\n"
    ),
    re.compile(
        rf"^\[Task handed to you by {_NAME}, a teammate on {_NAME}'s team\. It "
        r"is yours now: they are not waiting on a report and cannot answer "
        r"follow-ups\. Take it to completion and tell the user the outcome "
        r"yourself\. If something only the user can provide is missing, ask "
        r"them\.\]\r?\n\r?\n"
    ),
)


def strip_teammate_preamble(text: str) -> str:
    """``text`` without a leading teammate preamble.

    Text is returned unchanged when it does not open with one, or when the
    preamble is all there is, so nothing a user typed is ever emptied.
    """
    for pattern in _PREAMBLES:
        match = pattern.match(text)
        if match is None:
            continue
        rest = text[match.end() :]
        return rest if rest.strip() else text
    return text
