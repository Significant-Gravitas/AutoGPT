import pytest

from backend.copilot.capabilities import dispatch
from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.sources import tool_entries
from backend.copilot.tools import TOOL_REGISTRY


@pytest.mark.parametrize(
    "name",
    [
        "browser_request_link_payment",
        "browser_complete_link_payment",
        "browser_link_payment_status",
        "browser_reset_after_payment",
    ],
)
def test_checkout_capabilities_dispatch_through_the_registered_tools(monkeypatch, name):
    # A remote broker makes the checkout tools available on a self-hosted
    # deployment without the local runtime.
    monkeypatch.setenv("CHECKOUT_BROKER_URL", "https://broker:8443")
    tool = TOOL_REGISTRY[name]
    entries = tool_entries({name: tool}, {})
    assert len(entries) == 1
    assert not entries[0].eager
    registry = CapabilityIndex(entries)
    monkeypatch.setattr(dispatch, "get_registry", lambda: registry)
    arguments = {"checkout_id": "a" * 32}
    resolved = dispatch.resolve_tool_dispatch(
        "run_capability", {"id": f"tool:{name}", "input": arguments}
    )
    assert resolved is not None
    assert resolved.tool is tool
    assert resolved.name == name
    assert resolved.args == arguments
