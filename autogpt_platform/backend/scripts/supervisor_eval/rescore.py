"""Re-score harness JSONL files against the current actions.json, optionally on
one side of a split, so arms from different invocations sit in one report.

    rescore.py --out OUT.md [--split split.json --subset holdout] \\
        LABEL=file.jsonl[:model] [LABEL=file.jsonl[:model] ...]

Each LABEL renames the model column; ``:model`` picks one arm out of a file
that holds several (a Jev file, or a two-model run)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import measure  # noqa: E402

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=HERE / "actions.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--subset", choices=("all", "tune", "holdout"), default="all")
    parser.add_argument("arms", nargs="+", metavar="LABEL=file.jsonl[:model]")
    args = parser.parse_args()
    actions = measure.load_actions(args.actions)
    if args.split and args.subset != "all":
        keep = set(json.loads(args.split.read_text(encoding="utf-8"))[args.subset])
        actions = [a for a in actions if a.id in keep]
    ids = {a.id for a in actions}
    verdicts, models, runs = [], [], 1
    for arm in args.arms:
        label, path = arm.split("=", 1)
        path, _, only = path.partition(":")
        models.append(label)
        for line in open(path, encoding="utf-8"):
            v = measure.Verdict.model_validate_json(line)
            if v.rubric != "action" or v.item_id not in ids:
                continue
            if only and v.model != only:
                continue
            v.model = label
            runs = max(runs, v.run + 1)
            verdicts.append(v)
    result = measure.Run(models=models, runs=runs, actions=actions, verdicts=verdicts)
    scores = measure.score(result)
    report_args = argparse.Namespace(
        thinking="see arms",
        timeout=measure.GATE_TIMEOUT_S,
        subset=args.subset,
        answer_format="see arms",
    )
    report = measure.format_report(result, scores, report_args)
    args.out.write_text(report, encoding="utf-8")
    args.out.with_suffix(".scores.json").write_text(
        json.dumps(scores, indent=1, default=str), encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
