from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.api.features.builder import db
from backend.blocks._base import BlockInfo, BlockType


class _EmptyInputSchema:
    def get_credentials_fields_info(self):
        return {}

    def get_credentials_fields(self):
        return {}


def _block_info(block_id: str, name: str) -> BlockInfo:
    return BlockInfo(
        id=block_id,
        name=name,
        inputSchema={},
        outputSchema={},
        costs=[],
        description="",
        categories=[],
        contributors=[],
        staticOutput=False,
        uiType="",
    )


def _block(block_id: str, name: str, block_type: BlockType):
    class FakeBlock:
        id = block_id
        disabled = False
        categories = []
        input_schema = _EmptyInputSchema()

        def __init__(self):
            self.id = block_id
            self.name = name
            self.block_type = block_type
            self.disabled = False
            self.categories = []
            self.input_schema = _EmptyInputSchema()

        def get_info(self):
            return _block_info(block_id, name)

    return FakeBlock


@pytest.fixture
def mixed_blocks(monkeypatch):
    blocks = {
        "input": _block("input", "Input", BlockType.INPUT),
        "webhook": _block("webhook", "Webhook", BlockType.WEBHOOK),
        "webhook_manual": _block(
            "webhook_manual", "Webhook Manual", BlockType.WEBHOOK_MANUAL
        ),
        "output": _block("output", "Output", BlockType.OUTPUT),
        "standard": _block("standard", "Action", BlockType.STANDARD),
    }
    monkeypatch.setattr(db, "load_all_blocks", lambda: blocks)
    db._get_static_counts.cache_clear()
    return blocks


def test_get_blocks_treats_webhook_triggers_as_input_blocks(mixed_blocks):
    response = db.get_blocks(type="input")

    assert {block.id for block in response.blocks} == {
        "input",
        "webhook",
        "webhook_manual",
    }


def test_get_blocks_excludes_webhook_triggers_from_action_blocks(mixed_blocks):
    response = db.get_blocks(type="action")

    assert {block.id for block in response.blocks} == {"standard"}


@pytest.mark.asyncio
async def test_static_counts_classify_webhook_triggers_as_input_blocks(
    mixed_blocks, monkeypatch
):
    store_agent = SimpleNamespace(
        prisma=lambda: SimpleNamespace(count=AsyncMock(return_value=0))
    )
    monkeypatch.setattr(db.prisma.models, "StoreAgent", store_agent)

    counts = await db._get_static_counts()

    assert counts["input_blocks"] == 3
    assert counts["action_blocks"] == 1
    assert counts["output_blocks"] == 1
