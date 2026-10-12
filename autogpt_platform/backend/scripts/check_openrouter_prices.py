"""Compare the catalog's OpenRouter display prices against OpenRouter live.

``open_router`` catalog entries bill via ``COST_USD`` against the response's
``x-total-cost`` (see ``block_cost_config._open_router_llm_cost``), so their
``input_credits_per_1m`` / ``output_credits_per_1m`` are **display only** —
they feed ``_token_rate_display`` and the builder's "$X in / $Y out per 1M"
label, never a charge. OpenRouter reprices continuously, so these figures
drift by design; this script reports the drift instead of leaving the next
person to rediscover it.

Run manually::

    poetry run python scripts/check_openrouter_prices.py

Reads ``catalog.py`` with ``ast`` rather than importing the backend, so it
runs without a configured environment. A slug missing from
``/api/v1/models`` is not necessarily gone: OpenRouter keeps serving
renamed models that only the per-model ``/endpoints`` route lists, so
missing slugs are probed there before being called unavailable.

Exits 1 when any listed model's display rate differs from the live price.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# 100 credits/$ x 1.5 margin; mirrors block_cost_config._USD_PER_1M_DIVISOR.
CREDITS_PER_USD = 150
_MODELS_URL = "https://openrouter.ai/api/v1/models"
_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/%s/endpoints"
_CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "data"
    / "llm_registry"
    / "catalog.py"
)


def _literal(node: ast.AST | None) -> Any:
    try:
        return ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, TypeError, SyntaxError):
        return None


def catalog_models(path: Path) -> list[dict[str, Any]]:
    """Every ``open_router`` CatalogModel with its authored credit rates."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "CatalogModel"
        ):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if _literal(kw.get("provider")) != "open_router":
            continue
        entry: dict[str, Any] = {
            "slug": _literal(kw.get("slug")),
            "line": node.lineno,
            "input_credits_per_1m": None,
            "output_credits_per_1m": None,
        }
        cost = kw.get("cost")
        if isinstance(cost, ast.Call):
            ckw = {k.arg: k.value for k in cost.keywords}
            for field in ("input_credits_per_1m", "output_credits_per_1m"):
                entry[field] = _literal(ckw.get(field))
        out.append(entry)
    return out


def _probe_endpoints(slug: str) -> tuple[str, dict[str, Any] | None]:
    """Classify a slug that /api/v1/models does not list.

    ``REMOVED`` (404), ``ALIASED`` (renamed but still served — its pricing is
    returned), or ``NO_ENDPOINTS`` (known but nothing serving it).
    """
    try:
        with urllib.request.urlopen(_ENDPOINTS_URL % slug, timeout=60) as resp:
            data = json.load(resp)["data"]
    except urllib.error.HTTPError as exc:
        return ("REMOVED" if exc.code == 404 else "PROBE_ERROR"), None
    except (urllib.error.URLError, KeyError, ValueError):
        return "PROBE_ERROR", None
    endpoints = data.get("endpoints") or []
    if not endpoints:
        return "NO_ENDPOINTS", None
    cheapest = min(endpoints, key=lambda e: float(e["pricing"]["prompt"]))
    return "ALIASED", {"pricing": cheapest["pricing"], "alias_of": data.get("id")}


def compare(
    catalog_path: Path, models_json: Path | None = None
) -> list[dict[str, Any]]:
    if models_json is not None:
        live_raw = json.loads(models_json.read_text(encoding="utf-8"))["data"]
    else:
        with urllib.request.urlopen(_MODELS_URL, timeout=60) as resp:
            live_raw = json.load(resp)["data"]
    live = {m["id"]: m for m in live_raw}

    rows: list[dict[str, Any]] = []
    for model in catalog_models(catalog_path):
        row = dict(
            model,
            status="",
            alias_of=None,
            cat_input_usd=None,
            cat_output_usd=None,
            api_input_usd=None,
            api_output_usd=None,
            want_input_credits=None,
            want_output_credits=None,
        )
        if model["input_credits_per_1m"] is not None:
            row["cat_input_usd"] = model["input_credits_per_1m"] / CREDITS_PER_USD
            row["cat_output_usd"] = model["output_credits_per_1m"] / CREDITS_PER_USD

        api = live.get(model["slug"])
        aliased = False
        if api is None and models_json is None:
            status, alias = _probe_endpoints(model["slug"])
            row["status"] = status
            if alias is not None:
                row["alias_of"] = alias["alias_of"]
                api = alias
                aliased = True
        if api is None:
            row["status"] = row["status"] or "UNAVAILABLE"
            rows.append(row)
            continue

        pricing = api.get("pricing", {})
        row["api_input_usd"] = float(pricing.get("prompt", 0)) * 1e6
        row["api_output_usd"] = float(pricing.get("completion", 0)) * 1e6
        row["want_input_credits"] = round(row["api_input_usd"] * CREDITS_PER_USD, 4)
        row["want_output_credits"] = round(row["api_output_usd"] * CREDITS_PER_USD, 4)
        if model["input_credits_per_1m"] is None:
            row["status"] = "NO_RATES"
        elif (
            abs(row["cat_input_usd"] - row["api_input_usd"]) < 1e-9
            and abs(row["cat_output_usd"] - row["api_output_usd"]) < 1e-9
        ):
            row["status"] = "MATCH-ALIAS" if aliased else "MATCH"
        else:
            row["status"] = "DRIFT-ALIAS" if aliased else "DRIFT"
        rows.append(row)
    return rows


def _fmt(lo: float | None, hi: float | None) -> str:
    return "-" if lo is None else f"{lo:.6g} / {hi:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=_CATALOG_PATH)
    parser.add_argument(
        "--models-json",
        type=Path,
        help="use a saved /api/v1/models response instead of fetching",
    )
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args()

    rows = compare(args.catalog, args.models_json)
    rows.sort(key=lambda r: (r["status"], r["slug"]))

    width = max(len(r["slug"]) for r in rows) + 1
    print(
        f"{'slug':<{width}} {'status':<13} {'catalog $in/$out':>22} "
        f"{'live $in/$out':>22}  credits in/out (want)"
    )
    print("-" * (width + 94))
    for row in rows:
        want = (
            "-"
            if row["want_input_credits"] is None
            else f"{row['want_input_credits']:g} / {row['want_output_credits']:g}"
        )
        print(
            f"{row['slug']:<{width}} {row['status']:<13} "
            f"{_fmt(row['cat_input_usd'], row['cat_output_usd']):>22} "
            f"{_fmt(row['api_input_usd'], row['api_output_usd']):>22}  {want}"
        )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"\ntotal open_router entries: {len(rows)}")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    drifted = [r for r in rows if r["status"].startswith("DRIFT")]
    if drifted:
        print(
            f"\n{len(drifted)} model(s) drifted from OpenRouter's live price.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
