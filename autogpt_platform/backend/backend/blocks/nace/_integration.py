"""Opt-in live smoke check: three blocks and one image request, synthetic data only."""

import argparse
import asyncio
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values
from PIL import Image
from pydantic import SecretStr

from backend.blocks._base import Block
from backend.data.execution import ExecutionContext

from ._client import NoulQuestion, call_drex
from ._config import NaceCredentials
from .choice import NaceChoiceBlock
from .score import NaceScoreBlock
from .yes_no import NaceYesNoBlock

STATE = "The customer reports being charged twice for the same order."
OPTIONS = {"billing": "Charges and payments", "technical": "Software failures"}
LEVELS = [
    "No evidence of a duplicate charge.",
    "A duplicate charge is suspected but not explicitly reported.",
    "The customer explicitly reports a duplicate charge for one order.",
]
CASES: tuple[tuple[type[Block], dict[str, Any]], ...] = (
    (
        NaceChoiceBlock,
        {"question": "Which team should handle this?", "options": OPTIONS},
    ),
    (
        NaceScoreBlock,
        {"question": "How clear is the duplicate-charge report?", "levels": LEVELS},
    ),
    (
        NaceYesNoBlock,
        {"question": "Does this concern billing?", "data": {"ticket": "synthetic-1"}},
    ),
)
IMAGE_QUESTION = "Is Image 1 a single solid colour?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Read only NACE_API_KEY from this dotenv file.",
    )
    parser.add_argument(
        "--output", type=Path, help="Also save every emitted pin to a JSON report."
    )
    args = parser.parse_args()
    api_key = _read_key(args.env_file)
    if not api_key:
        parser.error("Set NACE_API_KEY or pass --env-file containing that key.")
    credentials = NaceCredentials(
        id="5ed76f4a-ccde-4758-8060-27d2a2718f01",
        provider="nace",
        api_key=SecretStr(api_key),
        title="Local nace integration check",
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
        return dotenv_values(env_file).get("NACE_API_KEY")
    return os.environ.get("NACE_API_KEY")


def _solid_png_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


async def _run_image_request(api_key: str) -> int:
    result = await call_drex(
        api_key,
        STATE,
        {"judgment": NoulQuestion(instructions=IMAGE_QUESTION)},
        [_solid_png_data_url()],
    )
    print("\n=== Image request (call_drex) ===", flush=True)
    print("request:", result.request, flush=True)
    print("response:", result.response, flush=True)
    print(
        f"usage: input_tokens={result.input_tokens} "
        f"output_tokens={result.output_tokens} request_id={result.request_id}",
        flush=True,
    )
    if result.error:
        raise RuntimeError("The image request returned an error.")
    return 1


async def _run_cases(credentials: NaceCredentials, output: Path | None) -> None:
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
        async for pin, value in block.run(
            input_data, credentials=credentials, execution_context=ExecutionContext()
        ):
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
    request_count += await _run_image_request(credentials.api_key.get_secret_value())
    if request_count != len(CASES) + 1:
        raise RuntimeError("Expected exactly four requests.")
    print(f"\nPASS: {len(CASES)} blocks, {request_count} requests.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
