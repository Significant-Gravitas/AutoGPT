"""Shared fakes and helpers for the SpecStream block tests."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from backend.blocks.specstream._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.specstream._models import Execution
from backend.blocks.specstream._testdata import (
    TEST_EXECUTION,
    TEST_FIND_ID,
    TEST_QUOTE_HASH,
)

QUOTE_BODY = {
    "quote_id": "qte_test_001",
    "quote_hash": TEST_QUOTE_HASH,
    "maximum_cents": 375,
    "expires_at": "2026-11-03T10:00:00Z",
}
ROWS_BODY = {
    "find_id": TEST_FIND_ID,
    "status": "completed",
    "accepted_count": 1,
    "next_cursor": None,
    "rows": [
        {
            "entity_id": "ent_1",
            "ordinal": 0,
            "cells": [
                {
                    "column_id": "identity",
                    "label": "Name",
                    "value": "Acme Pay",
                    "display": "Acme Pay",
                    "provenance": {
                        "source_url": "https://acme.example/about",
                        "observed_at": "2026-11-03T09:30:00Z",
                    },
                }
            ],
        }
    ],
}


class FakeResponse:
    """A stand-in for the platform's Response with a fixed status and JSON body."""

    def __init__(self, status_code: int, body: dict[str, Any]):
        """Store the status and the body as JSON text."""
        self.status = status_code
        self.ok = 200 <= status_code < 300
        self._text = json.dumps(body)

    def text(self) -> str:
        """Return the body as text, as Response.text does."""
        return self._text


def fake_requests(*responses: FakeResponse) -> MagicMock:
    """Return a Requests mock that answers with the given responses in order."""
    requests = MagicMock()
    requests.request = AsyncMock(side_effect=list(responses))
    return requests


async def run_block(block, **inputs) -> dict[str, Any]:
    """Run a block with the test credentials; return each output's first value."""
    outputs: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
        credentials=TEST_CREDENTIALS,
    ):
        outputs.setdefault(name, value)
    return outputs


def execution(find_id: str | None = TEST_FIND_ID) -> Execution:
    """Return the test execution with the given run ID."""
    return TEST_EXECUTION.model_copy(update={"find_id": find_id})


def approve_inputs(**extra) -> dict[str, Any]:
    """Return the Approve and Run inputs for the test quote, with any extra inputs."""
    return {
        "quote_id": "qte_test_001",
        "quote_hash": TEST_QUOTE_HASH,
        "confirm_max_cents": 375,
        **extra,
    }
