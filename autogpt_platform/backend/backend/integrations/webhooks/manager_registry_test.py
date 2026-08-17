"""Registry-wide invariants for webhook managers.

`BaseWebhooksManager` declares `validate_payload`, `_register_webhook` and
`_deregister_webhook` abstract. A manager that omits one still imports fine,
still registers against its provider, and still passes the block test harness —
Python only refuses at *instantiation*, and nothing instantiates a webhook
manager until a user actually sets up a trigger. The failure therefore lands on
the user as::

    Can't instantiate abstract class <X>WebhooksManager without an
    implementation for abstract method 'validate_payload'

These tests close that gap by exercising, for every registered provider, the
same call the runtime makes.
"""

from typing import Type

import pytest

from backend.blocks import get_blocks
from backend.blocks._base import Block
from backend.integrations.webhooks import (
    get_webhook_manager,
    load_webhook_managers,
    supports_webhooks,
)
from backend.integrations.webhooks._base import BaseWebhooksManager


@pytest.fixture(scope="module", autouse=True)
def _load_sdk_providers():
    """Import every block so SDK-registered webhook managers join the registry.

    Providers built with `ProviderBuilder(...).with_webhook_manager(...)` are
    only added to `load_webhook_managers()` once their block module is imported.
    """
    get_blocks()


def _webhook_blocks() -> list[Type[Block]]:
    return [
        block_cls
        for block_cls in get_blocks().values()
        if block_cls().webhook_config is not None
    ]


def test_registry_is_not_empty():
    # Guards against the fixture silently failing to import blocks, which would
    # make every other test here vacuously pass.
    assert load_webhook_managers()


def test_every_registered_manager_implements_all_abstract_methods():
    unimplemented = {
        provider: sorted(manager_cls.__abstractmethods__)
        for provider, manager_cls in load_webhook_managers().items()
        if getattr(manager_cls, "__abstractmethods__", frozenset())
    }

    assert not unimplemented, (
        "These webhook managers are still abstract and will raise TypeError the "
        "first time a user sets up a trigger: "
        f"{unimplemented}"
    )


def test_every_registered_manager_can_be_instantiated():
    # Snapshot the keys first: `get_webhook_manager` calls `load_webhook_managers`
    # again, and the SDK's patched loader re-inserts its managers into the same
    # cached dict, which would mutate it mid-iteration.
    failures: dict[str, str] = {}
    for provider in list(load_webhook_managers()):
        try:
            get_webhook_manager(provider)
        except Exception as exc:
            failures[str(provider)] = f"{type(exc).__name__}: {exc}"

    assert not failures, f"Webhook managers that fail to instantiate: {failures}"


def test_every_registered_manager_subclasses_the_base():
    wrong_base = {
        str(provider): manager_cls.__name__
        for provider, manager_cls in load_webhook_managers().items()
        if not issubclass(manager_cls, BaseWebhooksManager)
    }

    assert (
        not wrong_base
    ), f"Webhook managers not deriving from BaseWebhooksManager: {wrong_base}"


def _block_name(block_cls: Type[Block]) -> str:
    return block_cls().name


@pytest.mark.parametrize("block_cls", _webhook_blocks(), ids=_block_name)
def test_webhook_block_has_a_usable_manager(block_cls: Type[Block]):
    """A block advertising a webhook must have a manager that actually works.

    This is the end-to-end shape of the bug: the block registers, the provider
    registers, and only `get_webhook_manager()` — reached during trigger setup —
    discovers the manager is unusable.
    """
    block = block_cls()
    webhook_config = block.webhook_config
    assert webhook_config is not None

    provider = webhook_config.provider
    assert supports_webhooks(
        provider
    ), f"{block.name} declares webhook provider {provider!r} with no registered manager"

    get_webhook_manager(provider)
