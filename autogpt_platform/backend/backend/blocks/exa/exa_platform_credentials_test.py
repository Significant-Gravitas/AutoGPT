"""Exa blocks that keep objects in the key's account refuse the platform's key.

Websets, their searches, items, enrichments, monitors and imports, research
tasks and webhooks all live in the Exa account behind the key, so these blocks
only run on the user's own Exa key. Every module in ``backend.blocks.exa`` is
searched for blocks, so a block added later must either refuse the platform's
key or be named in ``STATELESS_BLOCKS``.
"""

import importlib
import inspect
import pkgutil
from enum import Enum
from pathlib import Path
from types import ModuleType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from backend.blocks._base import Block
from backend.blocks.exa._webhook import ExaWebhookManager
from backend.data.integrations import Webhook
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.credentials_store import exa_credentials
from backend.integrations.providers import ProviderName
from backend.util.exceptions import BlockExecutionError

STATELESS_BLOCKS = {
    "ExaAnswerBlock",
    "ExaCodeContextBlock",
    "ExaContentsBlock",
    "ExaFindSimilarBlock",
    "ExaPreviewWebsetBlock",
    "ExaSearchBlock",
}
# Blocks that refuse the platform's key but never call Exa themselves.
NO_VENDOR_CALL = {"ExaWebsetWebhookBlock"}
VENDOR_CLIENTS = ("AsyncExa", "Exa", "Requests")

SDK_DEFAULT = APIKeyCredentials(
    id="exa-default",
    provider="exa",
    api_key=SecretStr("platform-key"),
    title="Exa API Key",
)
USER_OWNED = APIKeyCredentials(
    id="0b0c6e43-5a1f-4a55-9a43-6f2b9f8d2a10",
    provider="exa",
    api_key=SecretStr("user-key"),
    title="My Exa key",
)
PLATFORM_KEYS = [exa_credentials, SDK_DEFAULT]


class ReachedVendor(BaseException):
    """Raised by the mocked vendor client. A BaseException, so a block's own
    ``except Exception`` can't swallow it and retry in a polling loop."""


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"backend.blocks.exa.{name}")


MODULES = [
    _module(info.name)
    for info in pkgutil.iter_modules([str(Path(__file__).parent)])
    if not info.name.startswith("_")
    and not info.name.endswith("_test")
    and info.name != "conftest"
]
ALL_BLOCKS = [
    cls
    for module in MODULES
    for _, cls in inspect.getmembers(module, inspect.isclass)
    if issubclass(cls, Block) and cls is not Block and cls.__module__ == module.__name__
]
STATEFUL_BLOCKS = [cls for cls in ALL_BLOCKS if cls.__name__ not in STATELESS_BLOCKS]
STATELESS_BLOCK_CLASSES = [
    cls for cls in ALL_BLOCKS if cls.__name__ in STATELESS_BLOCKS
]


def test_discovery_finds_the_blocks():
    names = {cls.__name__ for cls in STATEFUL_BLOCKS}
    assert {
        "ExaCreateWebsetBlock",
        "ExaListWebsetsBlock",
        "ExaDeleteWebsetBlock",
        "ExaCreateResearchBlock",
        "ExaListResearchBlock",
        "ExaUpdateEnrichmentBlock",
        "ExaListMonitorsBlock",
        "ExaExportWebsetBlock",
        "ExaWaitForEnrichmentBlock",
        "ExaWebsetWebhookBlock",
    } <= names
    assert len(STATEFUL_BLOCKS) >= 43
    assert {cls.__name__ for cls in STATELESS_BLOCK_CLASSES} == STATELESS_BLOCKS


@pytest.fixture
def vendor(monkeypatch: pytest.MonkeyPatch) -> list[MagicMock]:
    """Replace every Exa client in the Exa block modules with a mock that
    raises ``ReachedVendor`` on first use."""
    mocks = []
    for module in MODULES:
        for client in VENDOR_CLIENTS:
            if hasattr(module, client):
                mock = MagicMock(
                    side_effect=ReachedVendor(f"{module.__name__}.{client}")
                )
                monkeypatch.setattr(module, client, mock)
                mocks.append(mock)
    return mocks


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", PLATFORM_KEYS, ids=lambda c: c.id)
@pytest.mark.parametrize("block_cls", STATEFUL_BLOCKS, ids=lambda c: c.__name__)
async def test_stateful_block_refuses_platform_key(
    block_cls: type[Block], credentials: Credentials, vendor: list[MagicMock]
):
    block = block_cls()

    with pytest.raises(BlockExecutionError, match="platform's shared Exa key"):
        await _run(block, credentials)

    assert not any(mock.called for mock in vendor)


@pytest.mark.asyncio
@pytest.mark.parametrize("block_cls", STATEFUL_BLOCKS, ids=lambda c: c.__name__)
async def test_stateful_block_runs_on_users_own_key(
    block_cls: type[Block], vendor: list[MagicMock]
):
    if block_cls.__name__ in NO_VENDOR_CALL:
        await _run(block_cls(), USER_OWNED)
        return
    with pytest.raises(ReachedVendor):
        await _run(block_cls(), USER_OWNED)


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", PLATFORM_KEYS, ids=lambda c: c.id)
@pytest.mark.parametrize("block_cls", STATELESS_BLOCK_CLASSES, ids=lambda c: c.__name__)
async def test_stateless_block_still_runs_on_platform_key(
    block_cls: type[Block], credentials: Credentials, vendor: list[MagicMock]
):
    with pytest.raises(ReachedVendor):
        await _run(block_cls(), credentials)


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", PLATFORM_KEYS, ids=lambda c: c.id)
async def test_webhook_register_refuses_platform_key(
    credentials: Credentials, monkeypatch: pytest.MonkeyPatch
):
    requests = MagicMock(side_effect=ReachedVendor("Requests"))
    monkeypatch.setattr("backend.blocks.exa._webhook.Requests", requests)

    with pytest.raises(ValueError, match="platform's shared Exa key"):
        await ExaWebhookManager()._register_webhook(
            credentials, "webset", "", ["webset.created"], "https://ingress", "s"
        )

    requests.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_register_runs_on_users_own_key(
    monkeypatch: pytest.MonkeyPatch,
):
    requests = MagicMock(side_effect=ReachedVendor("Requests"))
    monkeypatch.setattr("backend.blocks.exa._webhook.Requests", requests)

    with pytest.raises(ReachedVendor):
        await ExaWebhookManager()._register_webhook(
            USER_OWNED, "webset", "", ["webset.created"], "https://ingress", "s"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", PLATFORM_KEYS, ids=lambda c: c.id)
async def test_webhook_registered_on_platform_key_can_still_be_deregistered(
    credentials: Credentials, monkeypatch: pytest.MonkeyPatch
):
    response = MagicMock(ok=True, status=200)
    delete = MagicMock()

    async def fake_delete(url: str, headers: dict) -> MagicMock:
        delete(url, headers)
        return response

    monkeypatch.setattr(
        "backend.blocks.exa._webhook.Requests",
        MagicMock(return_value=MagicMock(delete=fake_delete)),
    )
    webhook = Webhook(
        id="wh-1",
        user_id="user-a",
        provider=ProviderName("exa"),
        credentials_id=credentials.id,
        webhook_type="webset",
        resource="",
        events=["webset.created"],
        config={},
        secret="s",
        provider_webhook_id="exa-wh-1",
    )

    await ExaWebhookManager()._deregister_webhook(webhook, credentials)

    url, headers = delete.call_args.args
    assert url.endswith("/exa-wh-1")
    assert headers == {"x-api-key": credentials.api_key.get_secret_value()}


async def _run(block: Block, credentials: Credentials) -> None:
    async for _ in block.run(_sample_input(block), credentials=credentials):
        pass


def _sample_input(block: Block) -> Any:
    """Build an input without validation: defaults where the schema has them,
    plain sample values for required fields, and a URL for blocks that need
    one before calling Exa. The refusal must happen before any of it is read."""
    values = {
        name: (
            _sample(field.annotation)
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        for name, field in block.input_schema.model_fields.items()
        if name != "credentials"
    }
    if "urls" in values:
        values["urls"] = ["https://example.com"]
    return block.input_schema.model_construct(**values)


def _sample(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _sample(get_args(annotation)[0])
    if origin in (Union, UnionType):
        return _sample(next(a for a in get_args(annotation) if a is not type(None)))
    if origin is list:
        return []
    if origin is dict:
        return {}
    if inspect.isclass(annotation):
        if issubclass(annotation, Enum):
            return next(iter(annotation))
        if annotation is bool:
            return False
        if annotation is int:
            return 1
        if annotation is float:
            return 1.0
        if annotation is str:
            return "sample-id"
    return MagicMock()
