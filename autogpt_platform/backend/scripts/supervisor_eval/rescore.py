"""Re-score one or more harness JSONL files against the current actions.json.
usage: rescore.py ACTIONS.json OUT.md LABEL=file.jsonl [LABEL=file.jsonl ...]
Each LABEL renames the model column so arms from different invocations can sit in one report."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import measure  # noqa: E402

actions_path, out_path, *arms = sys.argv[1:]
actions = measure.load_actions(measure.Path(actions_path))
verdicts, models = [], []
runs = 1
for arm in arms:
    label, path = arm.split("=", 1)
    path, _, only = path.partition(":")
    models.append(label)
    for line in open(path):
        v = measure.Verdict.model_validate_json(line)
        if v.rubric != "action" or (only and v.model != only):
            continue
        v.model = label
        runs = max(runs, v.run + 1)
        verdicts.append(v)
result = measure.Run(models=models, runs=runs, actions=actions, verdicts=verdicts)
scores = measure.score(result)
args = argparse.Namespace(thinking="see arms", timeout=measure.GATE_TIMEOUT_S)
report = measure.format_report(result, scores, args)
open(out_path, "w").write(report)
json.dump(
    scores, open(out_path.replace(".md", ".scores.json"), "w"), indent=1, default=str
)
print(report)
