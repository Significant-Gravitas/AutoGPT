import pytest

from backend.util.request import validate_url


def test_validate_url():
    # Rejected IP ranges
    with pytest.raises(ValueError):
        validate_url("localhost", [])

    with pytest.raises(ValueError):
        validate_url("192.168.1.1", [])

    with pytest.raises(ValueError):
        validate_url("127.0.0.1", [])

    with pytest.raises(ValueError):
        validate_url("0.0.0.0", [])

    # Normal URLs
    assert validate_url("google.com/a?b=c", []) == "http://google.com/a?b=c"
    assert validate_url("github.com?key=!@!@", []) == "http://github.com?key=!@!@"

    # Scheme Enforcement
    with pytest.raises(ValueError):
        validate_url("ftp://example.com", [])
    with pytest.raises(ValueError):
        validate_url("file://example.com", [])

    # International domain that converts to punycode - should be allowed if public
    assert validate_url("http://xn--exmple-cua.com", []) == "http://xn--exmple-cua.com"
    # If the domain fails IDNA encoding or is invalid, it should raise an error
    with pytest.raises(ValueError):
        validate_url("http://exa◌mple.com", [])

    # IPv6 Addresses
    with pytest.raises(ValueError):
        validate_url("::1", [])  # IPv6 loopback should be blocked
    with pytest.raises(ValueError):
        validate_url("http://[::1]", [])  # IPv6 loopback in URL form

    # Suspicious Characters in Hostname
    with pytest.raises(ValueError):
        validate_url("http://example_underscore.com", [])
    with pytest.raises(ValueError):
        validate_url("http://exa mple.com", [])  # Space in hostname

    # Malformed URLs
    with pytest.raises(ValueError):
        validate_url("http://", [])  # No hostname
    with pytest.raises(ValueError):
        validate_url("://missing-scheme", [])  # Missing proper scheme

    # Trusted Origins
    trusted = ["internal-api.company.com", "10.0.0.5"]
    assert (
        validate_url("internal-api.company.com", trusted)
        == "http://internal-api.company.com"
    )
    assert validate_url("10.0.0.5", ["10.0.0.5"]) == "http://10.0.0.5"

    # Special Characters in Path or Query
    assert (
        validate_url("example.com/path%20with%20spaces", [])
        == "http://example.com/path%20with%20spaces"
    )

    # Backslashes should be replaced with forward slashes
    assert (
        validate_url("http://example.com\\backslash", [])
        == "http://example.com/backslash"
    )

    # Check defaulting scheme behavior for valid domains
    assert validate_url("example.com", []) == "http://example.com"
    assert validate_url("https://secure.com", []) == "https://secure.com"

    # Non-ASCII Characters in Query/Fragment
    assert validate_url("example.com?param=äöü", []) == "http://example.com?param=äöü"


@pytest.mark.parametrize(
    "hostname, resolved_ips, expect_error, expected_ip",
    [
        # Multiple public IPs, none blocked
        ("public-example.com", ["8.8.8.8", "9.9.9.9"], False, "8.8.8.8"),
        # Includes a blocked IP (e.g. link-local 169.254.x.x) => should raise
        ("rebinding.com", ["1.2.3.4", "169.254.169.254"], True, None),
        # Single public IP
        ("single-public.com", ["8.8.8.8"], False, "8.8.8.8"),
        # Single blocked IP
        ("blocked.com", ["127.0.0.1"], True, None),
    ],
)
def test_dns_rebinding_fix(
    monkeypatch, hostname, resolved_ips, expect_error, expected_ip
):
    """
    Tests that validate_url pins the first valid public IP address, and rejects
    the domain if any of the resolved IPs are blocked (i.e., DNS Rebinding scenario).
    """

    def mock_getaddrinfo(host, port, *args, **kwargs):
        # Simulate multiple IPs returned for the given hostname
        return [(None, None, None, None, (ip, port)) for ip in resolved_ips]

    # Patch socket.getaddrinfo so we control the DNS resolution in the test
    monkeypatch.setattr("socket.getaddrinfo", mock_getaddrinfo)

    if expect_error:
        # If any IP is blocked, we expect a ValueError
        with pytest.raises(ValueError):
            validate_url(hostname, [])
    else:
        pinned_url, ascii_hostname = validate_url(hostname, [])
        # The pinned_url should contain the first valid IP
        assert pinned_url.startswith("http://") or pinned_url.startswith("https://")
        assert expected_ip in pinned_url
        # The ascii_hostname should match our original hostname after IDNA encoding
        assert ascii_hostname == hostname
