"""Expert writing-style check (SECRT-2600).

Scores what each seeded expert would write for a fixed reference set of
prompts against that expert's own style specification, and reads the result
against the stored ``baseline.json``. Run by hand before shipping a prompt
or model-routing change — ``poetry run expert-style-eval --dry-run`` says
whether anything it measures has moved since the baseline.

The generation leg prompts the routed production model with the system
prompt production assembles — imported from the engines' own modules, never
retyped — so the score is about what production would say. The judge is a
cheap model reading a rubric shipped as data (``rubric.json``).
"""
