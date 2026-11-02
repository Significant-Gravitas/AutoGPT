"""Opt-in live smoke check: seven blocks, seven requests, synthetic data only."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values
from pydantic import SecretStr

from backend.blocks._base import Block
from backend.blocks.typesafe._config import TypeSafeCredentials
from backend.blocks.typesafe.ask_many import JevAskManyBlock
from backend.blocks.typesafe.choice import JevChoiceBlock
from backend.blocks.typesafe.filter import JevFilterBlock
from backend.blocks.typesafe.pick_best import JevPickBestBlock
from backend.blocks.typesafe.route import JevRouteBlock
from backend.blocks.typesafe.score import JevScoreBlock
from backend.blocks.typesafe.yes_no import JevYesNoBlock

STATE = "The customer reports being charged twice for the same order."
OPTIONS = {"billing": "Charges and payments", "technical": "Software failures"}
LEVELS = [
    "No evidence of a duplicate charge.",
    "A duplicate charge is suspected but not explicitly reported.",
    "The customer explicitly reports a duplicate charge for one order.",
]
CASES: tuple[tuple[type[Block], dict[str, Any]], ...] = (
    (
        JevChoiceBlock,
        {"question": "Which team should handle this?", "options": OPTIONS},
    ),
    (
        JevScoreBlock,
        {"question": "How clear is the duplicate-charge report?", "levels": LEVELS},
    ),
    (
        JevAskManyBlock,
        {
            "questions": {
                "team": {
                    "type": "choice",
                    "question": "Which team should handle this?",
                    "options": OPTIONS,
                },
                "evidence": {
                    "type": "score",
                    "question": "How clear is the duplicate-charge report?",
                    "levels": LEVELS,
                },
                "duplicate": {
                    "type": "noul",
                    "question": "Does the customer report a duplicate charge?",
                },
            }
        },
    ),
    (
        JevRouteBlock,
        {
            "question": "Which team should handle this?",
            "options": OPTIONS,
            "data": {"ticket": "synthetic-1"},
        },
    ),
    (
        JevYesNoBlock,
        {"question": "Does this concern billing?", "data": {"ticket": "synthetic-1"}},
    ),
    (
        JevPickBestBlock,
        {
            "question": "Which action best addresses the customer's report?",
            "candidates": [
                "Investigate the duplicate charge.",
                "Ask the customer to reinstall the app.",
            ],
        },
    ),
    (
        JevFilterBlock,
        {
            "items": ["The customer reports two charges for the same order."],
            "question": "How clear is the duplicate-charge report in the item?",
            "levels": LEVELS,
            "min_score": 1.0,
            "max_items": 1,
        },
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Read only TYPESAFE_API_KEY from this dotenv file.",
    )
    parser.add_argument(
        "--output", type=Path, help="Also save every emitted pin to a JSON report."
    )
    args = parser.parse_args()
    api_key = _read_key(args.env_file)
    if not api_key:
        parser.error("Set TYPESAFE_API_KEY or pass --env-file containing that key.")
    credentials = TypeSafeCredentials(
        id="5ed76f4a-ccde-4758-8060-27d2a2718f01",
        provider="typesafe",
        api_key=SecretStr(api_key),
        title="Local Jev integration check",
    )
    try:
        asyncio.run(_run_cases(credentials, args.output))
    except Exception as error:
        print(
            f"Integration check failed ({type(error).__name__}); "
            "exception details omitted to protect credentials.",
            file=sys.stderr,
        )
        return 1
    return 0


def _read_key(env_file: Path | None) -> str | None:
    if env_file is not None:
        if not env_file.is_file():
            raise SystemExit("The supplied dotenv file does not exist.")
        return dotenv_values(env_file).get("TYPESAFE_API_KEY")
    return os.environ.get("TYPESAFE_API_KEY")


async def _run_cases(credentials: TypeSafeCredentials, output: Path | None) -> None:
    metadata = {
        "id": credentials.id,
        "provider": credentials.provider,
        "type": credentials.type,
        "title": credentials.title,
    }
    request_count = 0
    reports = []
    for block_type, fields in CASES:
        block = block_type()
        input_data = block.input_schema.model_validate(
            {"state": STATE, "credentials": metadata, **fields}
        )
        print(f"\n=== {block_type.__name__} ===", flush=True)
        events = []
        async for pin, value in block.run(input_data, credentials=credentials):
            events.append({"pin": pin, "value": value})
            print(f"{pin}:", flush=True)
            if pin in {"request", "response"}:
                print(cast(str, value), flush=True)
            else:
                print(json.dumps(value, ensure_ascii=False, default=str), flush=True)
            request_count += pin == "request"
        reports.append({"block": block_type.__name__, "outputs": events})
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if any(event["pin"] == "error" for event in events):
            raise RuntimeError(f"{block_type.__name__} returned an error.")
    if request_count != len(CASES):
        raise RuntimeError("Expected exactly one request per block.")
    print(f"\nPASS: {len(CASES)} blocks, {request_count} requests.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
