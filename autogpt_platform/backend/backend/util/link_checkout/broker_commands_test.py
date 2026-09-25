import pytest

from backend.util.link_checkout.broker_commands import validate_command


@pytest.mark.parametrize(
    "args",
    [
        ["eval", "document.cookie"],
        ["get", "cdp-url"],
        ["cookies"],
        ["state", "save", "/tmp/state"],
        ["open", "file:///etc/passwd"],
        ["open", "http://shop.example"],
        ["open", "https://shop.example:8443"],
        ["open", "javascript:alert(1)"],
        ["fill", "--cdp", "9222"],
        ["click", "--config=/tmp/injected"],
        ["screenshot", "/tmp/output"],
        ["press", "Control+Shift+J"],
        ["screenshot", "--stream"],
        ["get", "url", "--headers=x"],
    ],
)
def test_broker_rejects_unrestricted_browser_channels(args):
    with pytest.raises(ValueError):
        validate_command(args)


@pytest.mark.parametrize(
    "args",
    [
        ["open", "https://shop.example"],
        ["fill", "@e3", "a street"],
        ["snapshot", "-i", "-c"],
        ["get", "url"],
        ["wait", "--load", "networkidle"],
        ["press", "Enter"],
    ],
)
def test_broker_accepts_bounded_navigation_controls(args):
    validate_command(args)
