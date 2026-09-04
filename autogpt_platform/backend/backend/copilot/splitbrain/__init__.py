"""EXPERIMENT — a split-brain prototype for AutoPilot, not production code.

AutoPilot today reasons and acts in one transcript, so every block schema it
reads stays in front of it for the rest of the run. This package runs the
alternative for real, on real AutoGPT block data, and measures it: a REASONER
transcript that holds the goal and can only dispatch intents, and an EXECUTOR
transcript that holds the tools and can only report back.

Written to answer whether the split is worth building, not to be shipped.
"""
