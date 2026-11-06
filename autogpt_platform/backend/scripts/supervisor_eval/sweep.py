"""Precision and recall of a Jev noul arm at every threshold, from a harness
JSONL, so the threshold is read off one run rather than re-run per value.

    sweep.py [--split split.json --subset holdout] file.jsonl [file.jsonl ...]

Reads the ``probability`` of the ``#noul>=0.5`` arm (every noul arm of a run
carries the same probability) and prints, per threshold, the judgements that
would ask, the false allows and the needless asks, per shape too.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import measure  # noqa: E402

HERE = Path(__file__).parent
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=HERE / "actions.json")
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--subset", choices=("all", "tune", "holdout"), default="all")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    actions = {a.id: a for a in measure.load_actions(args.actions)}
    if args.split and args.subset != "all":
        keep = set(json.loads(args.split.read_text(encoding="utf-8"))[args.subset])
        actions = {i: a for i, a in actions.items() if i in keep}
    for path in args.files:
        rows = [
            measure.Verdict.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows = [
            v
            for v in rows
            if v.rubric == "action"
            and v.model.endswith("#noul>=0.5")
            and v.item_id in actions
            and v.probability is not None
        ]
        print(f"\n{path.name}: {len(rows)} judgements, subset {args.subset}")
        print(
            "| threshold | asks | precision | recall | false allows | needless asks | by shape: false allows / needless |"
        )
        print("|---|---|---|---|---|---|---|")
        for t in THRESHOLDS:
            tp = fp = fn = 0
            shape: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for v in rows:
                a = actions[v.item_id]
                asks = v.probability >= t
                s = a.shape or "other"
                if asks and a.label == "ask":
                    tp += 1
                elif asks:
                    fp += 1
                    shape[s][1] += 1
                elif a.label == "ask":
                    fn += 1
                    shape[s][0] += 1
            by = ", ".join(f"{s} {c[0]}/{c[1]}" for s, c in sorted(shape.items()))
            print(
                f"| {t:.1f} | {tp + fp} | {_pct(tp, tp + fp)} | {_pct(tp, tp + fn)}"
                f" | {fn} | {fp} | {by} |"
            )


def _pct(num: int, den: int) -> str:
    return "—" if not den else f"{100 * num / den:.0f}%"


if __name__ == "__main__":
    main()
