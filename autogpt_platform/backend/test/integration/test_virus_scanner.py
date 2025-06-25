import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.server.v2.store.exceptions import VirusDetectedError, VirusScanError
from backend.util.virus_scanner import (
    VirusScannerService,
    VirusScannerSettings,
    VirusScanResult,
    get_virus_scanner,
    scan_content_safe,
)


class TestVirusScannerService:
    @pytest.fixture
    def scanner_settings(self):
        return VirusScannerSettings(
            clamav_service_host="localhost",
            clamav_service_port=3310,
            clamav_service_enabled=True,
            max_scan_size=10 * 1024 * 1024,  # 10MB for testing
        )

    @pytest.fixture
    def scanner(self, scanner_settings):
        return VirusScannerService(scanner_settings)

    @pytest.fixture
    def disabled_scanner(self):
        settings = VirusScannerSettings(clamav_service_enabled=False)
        return VirusScannerService(settings)

    def test_scanner_initialization(self, scanner_settings):
        scanner = VirusScannerService(scanner_settings)
        assert scanner.settings.clamav_service_host == "localhost"
        assert scanner.settings.clamav_service_port == 3310
        assert scanner.settings.clamav_service_enabled is True

    @pytest.mark.asyncio
    async def test_scan_disabled_returns_clean(self, disabled_scanner):
        content = b"test file content"
        result = await disabled_scanner.scan_file(content, filename="test.txt")

        assert result.is_clean is True
        assert result.threat_name is None
        assert result.file_size == len(content)
        assert result.scan_time_ms == 0

    @pytest.mark.asyncio
    async def test_scan_file_too_large(self, scanner):
        # Create content larger than max_scan_size
        large_content = b"x" * (scanner.settings.max_scan_size + 1)

        # Large files are allowed but marked as clean with a warning
        result = await scanner.scan_file(large_content, filename="large_file.txt")
        assert result.is_clean is True
        assert result.file_size == len(large_content)
        assert result.scan_time_ms == 0

    # Note: ping method was removed from current implementation

    @pytest.mark.asyncio
    @patch("pyclamd.ClamdNetworkSocket")
    async def test_scan_clean_file(self, mock_clamav_class, scanner):
        def mock_scan_stream(_):
            time.sleep(0.001)  # Small delay to ensure timing > 0
            return None  # No virus detected

        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.scan_stream = mock_scan_stream
        mock_clamav_class.return_value = mock_client

        content = b"clean file content"
        result = await scanner.scan_file(content, filename="clean.txt")

        assert result.is_clean is True
        assert result.threat_name is None
        assert result.file_size == len(content)
        assert result.scan_time_ms > 0

    @pytest.mark.asyncio
    @patch("pyclamd.ClamdNetworkSocket")
    async def test_scan_infected_file(self, mock_clamav_class, scanner):
        def mock_scan_stream(_):
            time.sleep(0.001)  # Small delay to ensure timing > 0
            return {"stream": ("FOUND", "Win.Test.EICAR_HDB-1")}

        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.scan_stream = mock_scan_stream
        mock_clamav_class.return_value = mock_client

        content = b"infected file content"
        result = await scanner.scan_file(content, filename="infected.txt")

        assert result.is_clean is False
        assert result.threat_name == "Win.Test.EICAR_HDB-1"
        assert result.file_size == len(content)
        assert result.scan_time_ms > 0

    @pytest.mark.asyncio
    @patch("pyclamd.ClamdNetworkSocket")
    async def test_scan_clamav_unavailable_fail_safe(self, mock_clamav_class, scanner):
        mock_client = Mock()
        mock_client.ping.return_value = False
        mock_clamav_class.return_value = mock_client

        content = b"test content"

        with pytest.raises(RuntimeError, match="ClamAV service is unreachable"):
            await scanner.scan_file(content, filename="test.txt")

    @pytest.mark.asyncio
    @patch("pyclamd.ClamdNetworkSocket")
    async def test_scan_error_fail_safe(self, mock_clamav_class, scanner):
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.scan_stream.side_effect = Exception("Scanning error")
        mock_clamav_class.return_value = mock_client

        content = b"test content"

        with pytest.raises(Exception, match="Scanning error"):
            await scanner.scan_file(content, filename="test.txt")

    # Note: scan_file_method and scan_upload_file tests removed as these APIs don't exist in current implementation

    def test_get_virus_scanner_singleton(self):
        scanner1 = get_virus_scanner()
        scanner2 = get_virus_scanner()

        # Should return the same instance
        assert scanner1 is scanner2

    # Note: client_reuse test removed as _get_client method doesn't exist in current implementation

    def test_scan_result_model(self):
        # Test VirusScanResult model
        result = VirusScanResult(
            is_clean=False, threat_name="Test.Virus", scan_time_ms=150, file_size=1024
        )

        assert result.is_clean is False
        assert result.threat_name == "Test.Virus"
        assert result.scan_time_ms == 150
        assert result.file_size == 1024

    @pytest.mark.asyncio
    @patch("pyclamd.ClamdNetworkSocket")
    async def test_concurrent_scans(self, mock_clamav_class, scanner):
        def mock_scan_stream(_):
            time.sleep(0.001)  # Small delay to ensure timing > 0
            return None

        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.scan_stream = mock_scan_stream
        mock_clamav_class.return_value = mock_client

        content1 = b"file1 content"
        content2 = b"file2 content"

        # Run concurrent scans
        results = await asyncio.gather(
            scanner.scan_file(content1, filename="file1.txt"),
            scanner.scan_file(content2, filename="file2.txt"),
        )

        assert len(results) == 2
        assert all(result.is_clean for result in results)
        assert results[0].file_size == len(content1)
        assert results[1].file_size == len(content2)
        assert all(result.scan_time_ms > 0 for result in results)


class TestHelperFunctions:
    """Test the helper functions scan_content_safe"""

    @pytest.mark.asyncio
    async def test_scan_content_safe_clean(self):
        """Test scan_content_safe with clean content"""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.return_value = Mock(
                is_clean=True, threat_name=None, scan_time_ms=50, file_size=100
            )
            mock_get_scanner.return_value = mock_scanner

            # Should not raise any exception
            await scan_content_safe(b"clean content", filename="test.txt")

    @pytest.mark.asyncio
    async def test_scan_content_safe_infected(self):
        """Test scan_content_safe with infected content"""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.return_value = Mock(
                is_clean=False, threat_name="Test.Virus", scan_time_ms=50, file_size=100
            )
            mock_get_scanner.return_value = mock_scanner

            with pytest.raises(VirusDetectedError) as exc_info:
                await scan_content_safe(b"infected content", filename="virus.txt")

            assert exc_info.value.threat_name == "Test.Virus"

    @pytest.mark.asyncio
    async def test_scan_content_safe_scan_error(self):
        """Test scan_content_safe when scanning fails"""
        with patch("backend.util.virus_scanner.get_virus_scanner") as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.scan_file = AsyncMock()
            mock_scanner.scan_file.side_effect = Exception("Scan failed")
            mock_get_scanner.return_value = mock_scanner

            with pytest.raises(VirusScanError, match="Virus scanning failed"):
                await scan_content_safe(b"test content", filename="test.txt")
