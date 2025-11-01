import pytest
from aiohttp import web

from backend.util.request import pin_url, validate_url


@pytest.mark.parametrize(
    "raw_url, trusted_origins, expected_value, should_raise",
    [
        # Rejected IP ranges
        ("localhost", [], None, True),
        ("192.168.1.1", [], None, True),
        ("127.0.0.1", [], None, True),
        ("0.0.0.0", [], None, True),
        # Normal URLs (should default to http:// if no scheme provided)
        ("google.com/a?b=c", [], "http://google.com/a?b=c", False),
        ("github.com?key=!@!@", [], "http://github.com?key=!@!@", False),
        # Scheme Enforcement
        ("ftp://example.com", [], None, True),
        ("file://example.com", [], None, True),
        # International domain converting to punycode (allowed if public)
        ("http://xn--exmple-cua.com", [], "http://xn--exmple-cua.com", False),
        # Invalid domain (IDNA failure)
        ("http://exa◌mple.com", [], None, True),
        # IPv6 addresses (loopback/blocked)
        ("::1", [], None, True),
        ("http://[::1]", [], None, True),
        # Suspicious Characters in Hostname
        ("http://example_underscore.com", [], None, True),
        ("http://exa mple.com", [], None, True),
        # Malformed URLs
        ("http://", [], None, True),  # No hostname
        ("://missing-scheme", [], None, True),  # Missing proper scheme
        # Trusted Origins
        (
            "internal-api.company.com",
            ["internal-api.company.com", "10.0.0.5"],
            "http://internal-api.company.com",
            False,
        ),
        ("10.0.0.5", ["10.0.0.5"], "http://10.0.0.5", False),
        # Special Characters in Path
        (
            "example.com/path%20with%20spaces",
            [],
            "http://example.com/path%20with%20spaces",
            False,
        ),
        # Backslashes should be replaced with forward slashes
        ("http://example.com\\backslash", [], "http://example.com/backslash", False),
        # Check default-scheme behavior for valid domains
        ("example.com", [], "http://example.com", False),
        ("https://secure.com", [], "https://secure.com", False),
        # Non-ASCII Characters in Query/Fragment
        ("example.com?param=äöü", [], "http://example.com?param=äöü", False),
    ],
)
async def test_validate_url_no_dns_rebinding(
    raw_url: str, trusted_origins: list[str], expected_value: str, should_raise: bool
):
    if should_raise:
        with pytest.raises(ValueError):
            await validate_url(raw_url, trusted_origins)
    else:
        validated_url, _, _ = await validate_url(raw_url, trusted_origins)
        assert validated_url.geturl() == expected_value


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
async def test_dns_rebinding_fix(
    monkeypatch,
    hostname: str,
    resolved_ips: list[str],
    expect_error: bool,
    expected_ip: str,
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
            url, _, ip_addresses = await validate_url(hostname, [])
            pin_url(url, ip_addresses)
    else:
        url, _, ip_addresses = await validate_url(hostname, [])
        pinned_url = pin_url(url, ip_addresses).geturl()
        # The pinned_url should contain the first valid IP
        assert pinned_url.startswith("http://") or pinned_url.startswith("https://")
        assert expected_ip in pinned_url
        # The unpinned URL's hostname should match our original IDNA encoded hostname
        assert url.hostname == hostname


@pytest.mark.asyncio
async def test_large_header_handling():
    """Test that ClientSession with max_field_size=16384 can handle large headers (>8190 bytes)"""
    import aiohttp

    # Create a test server that returns large headers
    async def large_header_handler(request):
        # Create a header value larger than the default aiohttp max_field_size (8190 bytes)
        # Simulate a long CSP header or similar legitimate large header
        large_value = "policy-" + "x" * 8500
        return web.Response(
            text="OK",
            headers={"X-Large-Header": large_value},
        )

    app = web.Application()
    app.router.add_get("/large-header", large_header_handler)

    # Start test server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    try:
        # Get the port from the server
        server = site._server
        assert server is not None
        sockets = getattr(server, "sockets", None)
        assert sockets is not None
        port = sockets[0].getsockname()[1]

        # Test with default max_field_size (should fail)
        default_failed = False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/large-header") as resp:
                    await resp.read()
        except Exception:
            # Expected: any error with default settings when header > 8190 bytes
            default_failed = True

        assert default_failed, "Expected error with default max_field_size"

        # Test with increased max_field_size (should succeed)
        # This is the fix: setting max_field_size=16384 allows headers up to 16KB
        async with aiohttp.ClientSession(max_field_size=16384) as session:
            async with session.get(f"http://127.0.0.1:{port}/large-header") as resp:
                body = await resp.read()
                # Verify the response is successful
                assert resp.status == 200
                assert "X-Large-Header" in resp.headers
                # Verify the large header value was received
                assert len(resp.headers["X-Large-Header"]) > 8190
                assert body == b"OK"

    finally:
        await runner.cleanup()
