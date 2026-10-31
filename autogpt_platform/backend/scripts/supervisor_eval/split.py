"""Write a TUNE/HOLDOUT split of actions.json: per (effect, label, shape) stratum,
shuffled with the seed, the first round(n/3) ids go to HOLDOUT.  Prints how
many items of each shape and label landed on each side.

    split.py --seed 92502 --written 2026-09-25 [--out split.json]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--written", required=True, help="date, for the record")
    parser.add_argument("--actions", type=Path, default=HERE / "actions.json")
    parser.add_argument("--out", type=Path, default=HERE / "split.json")
    args = parser.parse_args()
    items = json.loads(args.actions.read_text(encoding="utf-8"))["items"]
    strata: dict[tuple[str, str, str], list[str]] = {}
    for it in items:
        key = (it["effect"], it["label"], it.get("shape") or "")
        strata.setdefault(key, []).append(it["id"])
    rng = random.Random(args.seed)
    tune: list[str] = []
    holdout: list[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        k = round(len(ids) / 3)
        holdout += ids[:k]
        tune += ids[k:]
    order = {it["id"]: n for n, it in enumerate(items)}
    tune.sort(key=order.__getitem__)
    holdout.sort(key=order.__getitem__)
    out = {
        "seed": args.seed,
        "method": "stratified by (effect, label, shape); per stratum shuffled with"
        " random.Random(seed), the first round(n/3) to holdout",
        "written": args.written,
        "tune": tune,
        "holdout": holdout,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    by = {it["id"]: it for it in items}
    print(f"tune {len(tune)}, holdout {len(holdout)} (seed {args.seed})")
    print("| shape | label | tune | holdout |")
    print("|---|---|---|---|")
    keys = sorted({(by[i].get("shape") or "other", by[i]["label"]) for i in by})
    t = Counter((by[i].get("shape") or "other", by[i]["label"]) for i in tune)
    h = Counter((by[i].get("shape") or "other", by[i]["label"]) for i in holdout)
    for key in keys:
        print(f"| {key[0]} | {key[1]} | {t[key]} | {h[key]} |")
    for side, ids in (("tune", tune), ("holdout", holdout)):
        c = Counter((by[i]["effect"], by[i]["label"]) for i in ids)
        print(side, dict(sorted(c.items())))


if __name__ == "__main__":
    main()
