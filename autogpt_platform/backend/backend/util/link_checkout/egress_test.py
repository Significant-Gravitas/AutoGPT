import pytest

from backend.util.link_checkout.egress import destination, public_addresses


@pytest.mark.parametrize(
    "authority",
    [
        "shop.example:80",
        "shop.example:22",
        "evil.example:443",
        "shop.example.:443",
        "shop.example@evil.example:443",
        "127.0.0.1:443",
        "[::1]:443",
        "shop.example/path:443",
        "shop.example%00:443",
    ],
)
def test_only_explicit_https_hosts_can_receive_browser_traffic(authority):
    with pytest.raises(ValueError):
        destination(authority, {"shop.example", "127.0.0.1"})


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.0.0.1"],
        ["169.254.169.254"],
        ["::1"],
        ["fc00::1"],
        ["::ffff:127.0.0.1"],
        ["64:ff9b::7f00:1"],
        ["2002:7f00:1::"],
        ["2001:0:4136:e378:8000:63bf:3fff:fdd2"],
        ["93.184.216.34", "192.168.1.1"],
        [],
    ],
)
def test_any_private_dns_answer_denies_the_entire_connection(addresses):
    with pytest.raises(ValueError):
        public_addresses(addresses)


def test_public_address_is_resolved_once_and_pinned_for_connect():
    assert destination("shop.example:443", {"shop.example"}) == "shop.example"
    assert public_addresses(["93.184.216.34", "93.184.216.34"]) == ["93.184.216.34"]
