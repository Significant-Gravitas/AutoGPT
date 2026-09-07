"""Expert writing-style gate (SECRT-2600).

Scores what each seeded expert would write for a fixed reference set of
prompts against that expert's own style specification, and fails when an
expert's score drops below the agreed threshold. Runs from CI on changes to
the prompt-bearing and model-routing files (``assembly.TRIGGER_PATHS``).

The generation leg prompts the routed production model with the system
prompt production assembles — imported from the engines' own modules, never
retyped — so the score is about what production would say. The judge is a
cheap model reading a rubric shipped as data (``rubric.json``).
"""
