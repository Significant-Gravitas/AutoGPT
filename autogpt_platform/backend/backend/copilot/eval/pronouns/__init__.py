"""User-pronoun eval (SECRT-3161).

Otto and the experts must call the user "you", their name, or "they" when the
name is unknown, and never guess a gender. ``cases.py`` holds the personas and
the deterministic check; ``runner.py`` produces the text to check with the live
routed model and exits non-zero when any case uses he/she for the user.
"""
