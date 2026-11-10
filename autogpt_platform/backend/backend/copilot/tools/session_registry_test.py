import gc
import weakref
from collections import OrderedDict

import pytest

import backend.copilot.tools.session_registry as registry_module
from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.mcp_connections import custom_mcp_entry
from backend.copilot.capabilities.models import CapabilityEntry


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(registry_module, "_layered", OrderedDict())
    monkeypatch.setattr(registry_module, "_LAYERED_MAX", 2)


def _skill(owner: str) -> CapabilityEntry:
    return CapabilityEntry(
        id=f"skill:{owner}_private_skill",
        kind="skill",
        name=f"{owner}_private_skill",
        purpose=f"Private skill belonging to {owner}.",
    )


def test_mcp_layer_retains_evicted_skills_base_until_its_own_eviction():
    platform = CapabilityIndex([])
    alice = registry_module.layered_index(platform, [_skill("alice")])
    alice_ref = weakref.ref(alice)
    server = custom_mcp_entry("https://mcp.example.com/mcp")
    assert server is not None
    registry_module.layered_index(alice, [server])
    del alice

    registry_module.layered_index(platform, [_skill("other")])
    gc.collect()
    assert alice_ref() is not None

    bob = registry_module.layered_index(platform, [_skill("bob")])
    bob_mcp = registry_module.layered_index(bob, [server])
    assert bob_mcp.get("skill:bob_private_skill") is not None
    assert bob_mcp.get("skill:alice_private_skill") is None
    gc.collect()
    assert alice_ref() is None


def test_equal_cache_keys_cannot_reuse_another_users_skills(
    monkeypatch: pytest.MonkeyPatch,
):
    # Simulate address reuse deterministically, without depending on the allocator.
    monkeypatch.setattr(registry_module, "id", lambda _: 1, raising=False)
    alice = CapabilityIndex([_skill("alice")])
    bob = CapabilityIndex([_skill("bob")])
    server = custom_mcp_entry("https://mcp.example.com/mcp")
    assert server is not None

    alice_mcp = registry_module.layered_index(alice, [server])
    bob_mcp = registry_module.layered_index(bob, [server])

    assert bob_mcp is not alice_mcp
    assert bob_mcp.get("skill:bob_private_skill") is not None
    assert bob_mcp.get("skill:alice_private_skill") is None
    assert registry_module.layered_index(bob, [server]) is bob_mcp
