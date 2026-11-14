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

import pytest

from backend.blocks import get_blocks
from backend.blocks._base import Block
from backend.integrations import webhooks
from backend.integrations.webhooks import get_webhook_manager, supports_webhooks
from backend.integrations.webhooks._base import BaseWebhooksManager


def load_webhook_managers():
    """Resolve the loader off the module at call time.

    `AutoRegistry.patch_integrations()` replaces the module attribute, so a
    `from ... import load_webhook_managers` binding taken at import time can
    keep pointing at the unpatched original and miss every SDK-registered
    manager — which would make these checks pass vacuously.
    """
    return webhooks.load_webhook_managers()


@pytest.fixture(scope="module", autouse=True)
def _load_sdk_providers():
    """Import every block so SDK-registered webhook managers join the registry.

    Providers built with `ProviderBuilder(...).with_webhook_manager(...)` are
    only added to `load_webhook_managers()` once their block module is imported.
    """
    get_blocks()


def _webhook_blocks() -> list[type[Block]]:
    return [
        block_cls
        for block_cls in get_blocks().values()
        if block_cls().webhook_config is not None
    ]


def _registered_providers() -> list:
    """Snapshot the provider keys for parametrization.

    Snapshotted because `get_webhook_manager` calls `load_webhook_managers`
    again, and the SDK's patched loader re-inserts its managers into the same
    cached dict — iterating it live would mutate it mid-iteration.
    """
    get_blocks()
    return list(load_webhook_managers())


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


@pytest.mark.parametrize("provider", _registered_providers(), ids=str)
def test_every_registered_manager_can_be_instantiated(provider):
    # Parametrized rather than looped so a failure names the offending provider
    # and keeps its original traceback, instead of being flattened into a dict.
    get_webhook_manager(provider)


def test_every_registered_manager_subclasses_the_base():
    wrong_base = {
        str(provider): manager_cls.__name__
        for provider, manager_cls in load_webhook_managers().items()
        if not issubclass(manager_cls, BaseWebhooksManager)
    }

    assert (
        not wrong_base
    ), f"Webhook managers not deriving from BaseWebhooksManager: {wrong_base}"


def _block_name(block_cls: type[Block]) -> str:
    return block_cls().name


@pytest.mark.parametrize("block_cls", _webhook_blocks(), ids=_block_name)
def test_webhook_block_has_a_usable_manager(block_cls: type[Block]):
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
