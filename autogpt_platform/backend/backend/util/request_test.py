import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.request import Requests, pin_url, validate_url, verify_ocsp_stapling


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


# OCSP Stapling Tests
@pytest.mark.asyncio
async def test_ocsp_stapling_valid_server():
    """Test HTTPS request to a server with valid OCSP stapling (e.g., https://www.google.com)"""
    # Google typically has OCSP stapling enabled
    try:
        await verify_ocsp_stapling("www.google.com", 443, timeout=10)
    except Exception as e:
        # If OCSP verification is not supported in this environment, that's okay
        if "not supported" in str(e):
            pytest.skip(f"OCSP verification not supported: {e}")
        else:
            # For now, we'll skip if no OCSP response since not all servers have it enabled
            if "No OCSP stapled response" in str(e):
                pytest.skip(f"Server doesn't have OCSP stapling enabled: {e}")
            else:
                raise


@pytest.mark.asyncio
async def test_ocsp_stapling_no_ocsp_server():
    """Test HTTPS request to a server without OCSP stapling and verify appropriate error handling"""
    with patch("socket.create_connection") as mock_conn:
        mock_sock = MagicMock()
        mock_ssl_sock = MagicMock()
        mock_ssl_sock.ocsp_response = None
        mock_ssl_sock.getpeercert_bin = MagicMock()

        mock_conn.return_value.__enter__.return_value = mock_sock

        with patch("ssl.SSLContext.wrap_socket", return_value=mock_ssl_sock):
            with pytest.raises(Exception) as excinfo:
                await verify_ocsp_stapling("example.com", 443, timeout=5)
            assert "No OCSP stapled response received from example.com" in str(
                excinfo.value
            )


@pytest.mark.asyncio
async def test_http_requests_without_ocsp():
    """Test that HTTP requests work without OCSP verification"""
    # HTTP URLs should not trigger OCSP verification
    url, is_trusted, _ = await validate_url("http://example.com", [], verify_ocsp=True)
    assert url.scheme == "http"
    assert url.hostname == "example.com"


@pytest.mark.asyncio
async def test_trusted_origins_bypass_ocsp():
    """Test that trusted origins bypass OCSP verification"""
    # Mock the verify_ocsp_stapling to ensure it's not called
    with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
        url, is_trusted, _ = await validate_url(
            "https://trusted.example.com", ["trusted.example.com"], verify_ocsp=True
        )
        assert is_trusted
        mock_verify.assert_not_called()


@pytest.mark.asyncio
async def test_ocsp_timeout():
    """Test OCSP timeout by setting a very short timeout value"""
    with patch("socket.create_connection") as mock_conn:
        # Simulate timeout by raising socket.timeout
        mock_conn.side_effect = asyncio.TimeoutError("Connection timed out")

        with pytest.raises(asyncio.TimeoutError):
            await verify_ocsp_stapling("slow.example.com", 443, timeout=0.001)


@pytest.mark.asyncio
async def test_ocsp_disabled():
    """Test with verify_ocsp=False to ensure OCSP can be disabled"""
    with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
        url, _, _ = await validate_url("https://example.com", [], verify_ocsp=False)
        mock_verify.assert_not_called()


@pytest.mark.asyncio
async def test_requests_with_ocsp_disabled():
    """Test Requests class with OCSP verification disabled"""
    with patch("backend.util.request._resolve_host", return_value=["93.184.216.34"]):
        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {}
            mock_response.read = AsyncMock(return_value=b"OK")
            mock_response.raise_for_status = MagicMock()

            mock_request.return_value.__aenter__.return_value = mock_response

            requests = Requests(verify_ocsp=False)
            response = await requests.get("https://example.com")
            assert response.status == 200


@pytest.mark.asyncio
async def test_redirect_with_ocsp():
    """Test redirect handling with OCSP verification enabled"""
    with patch("backend.util.request._resolve_host", return_value=["93.184.216.34"]):
        with patch("aiohttp.ClientSession.request") as mock_request:
            # First response is a redirect
            mock_redirect = MagicMock()
            mock_redirect.status = 302
            mock_redirect.headers = {"Location": "https://redirected.example.com"}
            mock_redirect.read = AsyncMock(return_value=b"")

            # Second response is success
            mock_success = MagicMock()
            mock_success.status = 200
            mock_success.headers = {}
            mock_success.read = AsyncMock(return_value=b"Success")
            mock_success.raise_for_status = MagicMock()

            mock_request.return_value.__aenter__.side_effect = [
                mock_redirect,
                mock_success,
            ]

            with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
                # Mock OCSP verification to pass
                mock_verify.return_value = None

                requests = Requests(verify_ocsp=True)
                response = await requests.get(
                    "https://example.com", allow_redirects=True
                )
                assert response.status == 200
                assert response.content == b"Success"

                # OCSP should be verified for both original and redirect URLs
                assert mock_verify.call_count == 2


@pytest.mark.asyncio
async def test_async_event_loop_not_blocked():
    """Verify that blocking operations don't freeze the event loop"""

    async def concurrent_task():
        await asyncio.sleep(0.1)
        return "completed"

    with patch("socket.create_connection") as mock_conn:
        mock_sock = MagicMock()
        mock_ssl_sock = MagicMock()
        mock_ssl_sock.ocsp_response = None
        mock_ssl_sock.getpeercert_bin = MagicMock()

        mock_conn.return_value.__enter__.return_value = mock_sock

        with patch("ssl.SSLContext.wrap_socket", return_value=mock_ssl_sock):
            # Run OCSP verification and concurrent task together
            tasks = [
                verify_ocsp_stapling("example.com", 443, timeout=5),
                concurrent_task(),
            ]

            # The concurrent task should complete even if OCSP verification fails
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # First result should be an exception from OCSP verification
            assert isinstance(results[0], Exception)
            # Second result should be from the concurrent task
            assert results[1] == "completed"


@pytest.mark.asyncio
async def test_ipv4_and_ipv6_addresses():
    """Test with both IPv4 and IPv6 addresses"""
    # Test with IPv4
    with patch("backend.util.request._resolve_host", return_value=["93.184.216.34"]):
        with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
            mock_verify.return_value = None
            url, _, ips = await validate_url(
                "https://example.com", [], verify_ocsp=True
            )
            assert "93.184.216.34" in ips

    # Test with IPv6
    with patch(
        "backend.util.request._resolve_host",
        return_value=["2606:2800:220:1:248:1893:25c8:1946"],
    ):
        with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
            mock_verify.return_value = None
            url, _, ips = await validate_url(
                "https://example.com", [], verify_ocsp=True
            )
            assert "2606:2800:220:1:248:1893:25c8:1946" in ips

    # Test with both IPv4 and IPv6
    with patch(
        "backend.util.request._resolve_host",
        return_value=["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"],
    ):
        with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
            mock_verify.return_value = None
            url, _, ips = await validate_url(
                "https://example.com", [], verify_ocsp=True
            )
            assert "93.184.216.34" in ips
            assert "2606:2800:220:1:248:1893:25c8:1946" in ips


@pytest.mark.asyncio
async def test_ocsp_error_messages():
    """Verify that error messages are clear when OCSP verification fails"""
    # Test various OCSP failure scenarios

    # 1. No OCSP response
    with patch("socket.create_connection") as mock_conn:
        mock_sock = MagicMock()
        mock_ssl_sock = MagicMock()
        mock_ssl_sock.ocsp_response = None
        mock_ssl_sock.getpeercert_bin = MagicMock()

        mock_conn.return_value.__enter__.return_value = mock_sock

        with patch("ssl.SSLContext.wrap_socket", return_value=mock_ssl_sock):
            with pytest.raises(Exception) as excinfo:
                await verify_ocsp_stapling("test.example.com", 443)
            assert "No OCSP stapled response received from test.example.com" in str(
                excinfo.value
            )

    # 2. OCSP not supported in environment
    with patch("socket.create_connection") as mock_conn:
        mock_sock = MagicMock()
        mock_ssl_sock = MagicMock()
        # Remove ocsp_response attribute entirely
        del mock_ssl_sock.ocsp_response

        mock_conn.return_value.__enter__.return_value = mock_sock

        with patch("ssl.SSLContext.wrap_socket", return_value=mock_ssl_sock):
            with pytest.raises(Exception) as excinfo:
                await verify_ocsp_stapling("test.example.com", 443)
            assert "OCSP verification not supported" in str(excinfo.value)

    # 3. Test validate_url OCSP error propagation
    with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
        mock_verify.side_effect = Exception("Custom OCSP error for testing")

        with pytest.raises(ValueError) as excinfo:
            await validate_url("https://failing.example.com", [], verify_ocsp=True)
        assert (
            "OCSP verification failed for failing.example.com: Custom OCSP error for testing"
            in str(excinfo.value)
        )


@pytest.mark.asyncio
async def test_ocsp_with_custom_port():
    """Test OCSP verification with custom HTTPS port"""
    with patch("backend.util.request.verify_ocsp_stapling") as mock_verify:
        mock_verify.return_value = None

        # Test with custom port in URL
        url, _, _ = await validate_url("https://example.com:8443", [], verify_ocsp=True)

        # Verify that the custom port was passed to OCSP verification
        mock_verify.assert_called_once_with("example.com", 8443, timeout=5)
