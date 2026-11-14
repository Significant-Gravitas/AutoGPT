"""Refresh ``backend/blocks/replicate/official_prices.json`` from Replicate.

Lists Replicate's official collection through its API (``REPLICATE_API_KEY``),
reads each model's prices from the ``billingConfig`` its replicate.com page
embeds, one page at a time, and maps each price tier onto the model's inputs.
A model the block can't price from that is listed under ``unpriced`` with the
reason, and the block refuses to run it. Runs no model and costs nothing.

Run manually: ``poetry run python scripts/refresh_replicate_prices.py``.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_OUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "blocks"
    / "replicate"
    / "official_prices.json"
)
_COLLECTION_URL = "https://api.replicate.com/v1/collections/official"
_PAGE_URL = "https://replicate.com/{model}"
_PAGE_GAP_SECONDS = 2.0
_USER_AGENT = "autogpt-replicate-prices/1.0"

# How a billed quantity is measured when the run doesn't report it.
_OUTPUT_COUNTS = {
    "image_output_count",
    "video_output_count",
    "audio_output_count",
    "generic_output_count",
}
_DURATIONS = {"video_output_duration_seconds", "audio_output_duration_seconds"}
_TEXT_INPUTS = {"token_input_count", "character_input_count"}
_FILE_INPUTS = {"image_input_count", "video_input_count", "audio_input_count"}
_DURATION_INPUTS = (
    "duration",
    "seconds",
    "duration_seconds",
    "duration_sec",
    "length",
    "num_seconds",
)
_MAX_TOKEN_INPUTS = (
    "max_tokens",
    "max_new_tokens",
    "max_completion_tokens",
    "max_output_tokens",
)
# A tier criterion's title, and the key a run reports its value under.
_CRITERION_KEYS = {
    "target resolution": "resolution_target",
    "model variant": "model_variant",
    "with audio": "with_audio",
    "second of output video": "video_output_duration_seconds",
}
_RESOLUTION_INPUTS = ("resolution", "size", "image_size", "target_resolution")
# Why a model billed by a quantity no input or output measures can't be priced.
_UNMEASURED = {
    "unspecified_billing_metric": "priced in units Replicate works out from a table "
    "on its model page, not a quantity a run reports",
    "page_input_count": "billed per page of its input document, which the block "
    "can't count",
    "training_step_count": "a trainer, billed per training step",
}
_UNITS = {
    "video_output_duration_seconds": "the second of output video",
    "audio_output_duration_seconds": "the second of output audio",
    "token_output_count": "the output token",
    "token_input_count": "the input token",
    "character_input_count": "the input character",
}
_PIXEL_REASON = (
    "billed by the megapixels or pixels of its images, which depend on image sizes "
    "the block doesn't see"
)


def main() -> int:
    api_key = os.environ.get("REPLICATE_API_KEY")
    if not api_key:
        print("Set REPLICATE_API_KEY to list the official collection", file=sys.stderr)
        return 1
    models = _fetch_collection(api_key)
    priced: dict[str, Any] = {}
    unpriced: dict[str, str] = {}
    for index, model in enumerate(models):
        name = f"{model['owner']}/{model['name']}"
        if index:
            time.sleep(_PAGE_GAP_SECONDS)
        billing = _billing_config(_fetch(_PAGE_URL.format(model=name)))
        price, reason = build_price(model, billing)
        if price is None:
            unpriced[name] = reason
        else:
            priced[name] = price
    out = {
        "read_on": datetime.date.today().isoformat(),
        "collection": sorted([*priced, *unpriced]),
        "priced": dict(sorted(priced.items())),
        "unpriced": dict(sorted(unpriced.items())),
    }
    _OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(priced)} priced, {len(unpriced)} unpriced", file=sys.stderr)
    return 0


def build_price(
    model: dict[str, Any], billing: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str]:
    """The model's price entry, or None and the reason it can't be priced."""
    tiers_in = [t for t in (billing or {}).get("current_tiers", []) if t.get("prices")]
    if not tiers_in:
        return None, "Replicate publishes no price for it"
    schema = _input_schema(model)
    props = schema.get("properties", {})

    tiers: list[dict[str, Any]] = []
    metrics: set[str] = set()
    for tier in tiers_in:
        usd: dict[str, float] = {}
        for item in tier["prices"]:
            per_unit = _usd_per_unit(item)
            if per_unit is None:
                return None, f"a price row the block can't read: {item.get('title')!r}"
            usd[item["metric"]] = per_unit
            metrics.add(item["metric"])
        for criteria in _map_criteria(tier.get("criteria", []), props):
            tiers.append({"criteria": criteria, "usd_per_unit": usd})

    quantities: dict[str, Any] = {}
    for metric in sorted(metrics):
        quantity, reason = _quantity(metric, props)
        if quantity is None:
            return None, reason
        quantities[metric] = quantity
    # A model that answers in text bills its file inputs as input tokens too.
    unpriced_inputs = (
        [k for k, p in props.items() if _is_file(p)]
        if metrics & _TEXT_INPUTS and not _outputs_files(model)
        else []
    )

    used = {c["input"] for t in tiers for c in t["criteria"] if c.get("input")}
    used |= {key for q in quantities.values() for key in q.get("inputs", [])}
    defaults = {
        key: props[key]["default"]
        for key in sorted(used)
        if key in props and isinstance(props[key].get("default"), (str, int, float))
    }
    price = {"tiers": tiers, "quantities": quantities, "defaults": defaults}
    if unpriced_inputs:
        price["unpriced_inputs"] = unpriced_inputs
    return price, ""


def _quantity(metric: str, props: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if metric == "run_count":
        return {"kind": "per_run"}, ""
    if metric == "run_time":
        return {"kind": "run_time"}, ""
    if metric in _OUTPUT_COUNTS:
        return {"kind": "outputs"}, ""
    if metric in _DURATIONS:
        return _bounded(props, _DURATION_INPUTS, metric)
    if metric == "token_output_count":
        return _bounded(props, _MAX_TOKEN_INPUTS, metric)
    if metric in _TEXT_INPUTS:
        keys = [k for k, p in props.items() if _is_text(p)]
        if not keys:
            return None, f"billed by {_unit(metric)} and has no text input to count"
        return {"kind": "input_text", "inputs": keys}, ""
    if metric in _FILE_INPUTS:
        keys = [k for k, p in props.items() if _is_file(p)]
        if not keys:
            return None, f"billed by {_unit(metric)} and has no file input to count"
        return {"kind": "input_files", "inputs": keys}, ""
    if "pixel" in metric:
        return None, _PIXEL_REASON
    return None, _UNMEASURED.get(
        metric, f"billed by {_unit(metric)}, which no input or output measures"
    )


def _bounded(
    props: dict[str, Any],
    candidates: tuple[str, ...],
    metric: str,
) -> tuple[dict[str, Any] | None, str]:
    keys = [k for k in candidates if k in props]
    caps = [c for c in (_maximum(props[k]) for k in keys) if c is not None]
    max_units = max(caps) if caps else None
    if not keys:
        what = (
            "caps its output" if metric == "token_output_count" else "sets its length"
        )
        return None, f"billed by {_unit(metric)}, and no input {what}"
    quantity: dict[str, Any] = {"kind": "input_value", "inputs": keys}
    if max_units is not None:
        quantity["max_units"] = max_units
    elif not any(isinstance(props[k].get("default"), (int, float)) for k in keys):
        quantity["required"] = True
    return quantity, ""


def _map_criteria(
    criteria: list[dict[str, Any]], props: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    """A tier's criteria, once per input value that selects it (a resolution
    can be several sizes); a criterion no input sets stays unmapped."""
    combos: list[list[dict[str, Any]]] = [[]]
    for criterion in criteria:
        if criterion.get("type") != "equals":
            combos = [combo + [{"value": criterion["value"]}] for combo in combos]
            continue
        base = {
            "key": _CRITERION_KEYS.get(criterion.get("title", "")),
            "value": criterion["value"],
        }
        options = [
            {**base, "input": key, "input_value": value}
            for key, value in _input_options(criterion, props)
        ] or [base]
        combos = [combo + [option] for combo in combos for option in options]
    return combos


def _input_options(
    criterion: dict[str, Any], props: dict[str, Any]
) -> list[tuple[str, Any]]:
    title, value = criterion.get("title", ""), criterion["value"]
    if value is None:
        return []
    text = str(value).lower()
    if title == "second of output video":
        return [(k, value) for k in _DURATION_INPUTS if k in props][:1]
    if title == "with audio" or text in ("with_audio", "without_audio"):
        if props.get("generate_audio", {}).get("type") == "boolean":
            return [("generate_audio", text in ("true", "with_audio"))]
        return []
    if text in ("draft", "base") and props.get("draft", {}).get("type") == "boolean":
        return [("draft", text == "draft")]
    if title == "target resolution":
        for key in _RESOLUTION_INPUTS:
            enum = props.get(key, {}).get("enum") or []
            exact = [v for v in enum if str(v).lower() == text]
            lines = _resolution_lines(text)
            sized = [v for v in enum if lines and _short_side(v) == lines]
            if exact or sized:
                return [(key, v) for v in exact or sized]
        return []
    # Any other criterion maps only onto the one input whose choices hold its
    # value, or end it ("sora-2-pro-high" is the "high" resolution).
    for same in (
        lambda v: str(v).lower() == text,
        lambda v: text.endswith("-" + str(v).lower()),
    ):
        matches = [
            (key, v)
            for key, prop in props.items()
            for v in prop.get("enum") or []
            if same(v)
        ]
        if len({key for key, _ in matches}) == 1:
            return matches
    return []


def _usd_per_unit(item: dict[str, Any]) -> float | None:
    match = re.fullmatch(
        r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", item.get("price", "").strip()
    )
    title = item.get("title", "")
    if not match or not title.startswith("per "):
        return None
    usd = float(match.group(1).replace(",", ""))
    if title.startswith("per thousand "):
        return usd / 1e3
    if title.startswith("per million "):
        return usd / 1e6
    if re.match(r"per [0-9]", title):
        return None
    return usd


def _unit(metric: str) -> str:
    return _UNITS.get(metric, metric.replace("_", " "))


def _input_schema(model: dict[str, Any]) -> dict[str, Any]:
    """The model's input schema, with each input's ``allOf``/``$ref`` choices inlined."""
    schemas = _schemas(model)
    schema = dict(schemas.get("Input", {}))
    schema["properties"] = {
        key: _resolve(prop, schemas)
        for key, prop in schema.get("properties", {}).items()
    }
    return schema


def _resolve(prop: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    resolved = {k: v for k, v in prop.items() if k != "allOf"}
    for part in prop.get("allOf", []):
        ref = part.get("$ref", "").rsplit("/", 1)[-1]
        resolved = {**schemas.get(ref, part), **resolved}
    return resolved


def _outputs_files(model: dict[str, Any]) -> bool:
    output = _schemas(model).get("Output", {})
    return _is_file(output)


def _schemas(model: dict[str, Any]) -> dict[str, Any]:
    version = model.get("latest_version") or {}
    components = (version.get("openapi_schema") or {}).get("components", {})
    return components.get("schemas", {})


def _maximum(prop: dict[str, Any]) -> float | None:
    if isinstance(prop.get("maximum"), (int, float)):
        return float(prop["maximum"])
    numbers = [_as_number(v) for v in prop.get("enum") or []]
    known = [n for n in numbers if n is not None]
    return max(known) if known else None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_text(prop: dict[str, Any]) -> bool:
    if prop.get("type") == "array":
        return not _is_file(prop)
    return prop.get("type") == "string" and "enum" not in prop and not _is_file(prop)


def _is_file(prop: dict[str, Any]) -> bool:
    items = prop.get("items") or {}
    return prop.get("format") == "uri" or items.get("format") == "uri"


def _short_side(value: Any) -> int | None:
    match = re.fullmatch(r"\s*(\d+)\s*[x*]\s*(\d+)\s*", str(value))
    return min(int(match.group(1)), int(match.group(2))) if match else None


def _resolution_lines(text: str) -> int | None:
    match = re.fullmatch(r"(\d+)p", text)
    return int(match.group(1)) if match else None


def _fetch_collection(api_key: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    url: str | None = _COLLECTION_URL
    while url:
        data = json.loads(_fetch(url, {"Authorization": f"Bearer {api_key}"}))
        models += data.get("models", [])
        url = data.get("next")
    return models


def _billing_config(page: str) -> dict[str, Any] | None:
    for match in re.finditer(
        r'<script[^>]*type="application/json"[^>]*>([\s\S]*?)</script>', page
    ):
        found = _find_key(_load(match.group(1)), "billingConfig")
        if found:
            return found
    return None


def _find_key(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if node.get(key):
            return node[key]
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            found = _find_key(child, key)
            if found:
                return found
    return None


def _load(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fetch(url: str, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


if __name__ == "__main__":
    raise SystemExit(main())
