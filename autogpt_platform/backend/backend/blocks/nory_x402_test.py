"""Tests for Nory x402 Payment Blocks."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.blocks.nory_x402 import (
    NoryGetPaymentRequirementsBlock,
    NoryVerifyPaymentBlock,
    NorySettlePaymentBlock,
    NoryTransactionLookupBlock,
    NoryHealthCheckBlock,
    NoryNetwork,
)


class MockResponse:
    """Mock HTTP response."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data


@pytest.fixture
def mock_requests():
    """Mock the Requests class."""
    with patch("backend.blocks.nory_x402.Requests") as mock:
        yield mock


class TestNoryGetPaymentRequirementsBlock:
    """Tests for NoryGetPaymentRequirementsBlock."""

    def test_block_init(self):
        """Test block initialization."""
        block = NoryGetPaymentRequirementsBlock()
        assert block.id == "2bd9a224-4bd9-4280-bd17-bbe6c970bc9a"
        assert "x402 payment requirements" in block.description.lower()

    @pytest.mark.asyncio
    async def test_run_success(self, mock_requests):
        """Test successful payment requirements request."""
        expected_response = {
            "x402Version": 2,
            "accepts": [
                {
                    "network": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
                    "amount": "1000000",
                    "currency": "USDC",
                }
            ],
        }

        mock_requests_instance = MagicMock()
        mock_requests_instance.get = AsyncMock(
            return_value=MockResponse(expected_response)
        )
        mock_requests.return_value = mock_requests_instance

        block = NoryGetPaymentRequirementsBlock()
        input_data = block.Input(
            resource="/api/premium/data",
            amount="0.10",
            network=NoryNetwork.SOLANA_MAINNET,
            api_key="test_key",
        )

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "requirements"
        assert results[0][1] == expected_response

    @pytest.mark.asyncio
    async def test_run_error(self, mock_requests):
        """Test error handling."""
        mock_requests_instance = MagicMock()
        mock_requests_instance.get = AsyncMock(side_effect=Exception("Network error"))
        mock_requests.return_value = mock_requests_instance

        block = NoryGetPaymentRequirementsBlock()
        input_data = block.Input(
            resource="/api/premium/data",
            amount="0.10",
        )

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "error"
        assert "Network error" in results[0][1]


class TestNoryVerifyPaymentBlock:
    """Tests for NoryVerifyPaymentBlock."""

    def test_block_init(self):
        """Test block initialization."""
        block = NoryVerifyPaymentBlock()
        assert block.id == "a8160ae9-876c-45b1-a23c-0c89608dcb01"

    @pytest.mark.asyncio
    async def test_run_success(self, mock_requests):
        """Test successful payment verification."""
        expected_response = {
            "valid": True,
            "payer": "5C1MADvYEtGA3ktUPWsT3RqmvAYuugi3tnqrUHdR5Rip",
            "amount": "1000000",
        }

        mock_requests_instance = MagicMock()
        mock_requests_instance.post = AsyncMock(
            return_value=MockResponse(expected_response)
        )
        mock_requests.return_value = mock_requests_instance

        block = NoryVerifyPaymentBlock()
        input_data = block.Input(
            payload="base64encodedpayload",
            api_key="test_key",
        )

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "result"
        assert results[0][1]["valid"] is True

    @pytest.mark.asyncio
    async def test_run_error(self, mock_requests):
        """Test error handling."""
        mock_requests_instance = MagicMock()
        mock_requests_instance.post = AsyncMock(
            side_effect=Exception("Verification failed")
        )
        mock_requests.return_value = mock_requests_instance

        block = NoryVerifyPaymentBlock()
        input_data = block.Input(payload="invalid_payload")

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "error"


class TestNorySettlePaymentBlock:
    """Tests for NorySettlePaymentBlock."""

    def test_block_init(self):
        """Test block initialization."""
        block = NorySettlePaymentBlock()
        assert block.id == "787dad1d-c64c-4996-89b6-8390b73b17f8"

    @pytest.mark.asyncio
    async def test_run_success(self, mock_requests):
        """Test successful payment settlement."""
        expected_response = {
            "success": True,
            "transactionId": "abc123",
            "settledAt": 1706745600000,
        }

        mock_requests_instance = MagicMock()
        mock_requests_instance.post = AsyncMock(
            return_value=MockResponse(expected_response)
        )
        mock_requests.return_value = mock_requests_instance

        block = NorySettlePaymentBlock()
        input_data = block.Input(
            payload="base64encodedpayload",
            api_key="test_key",
        )

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "result"
        assert results[0][1]["success"] is True


class TestNoryTransactionLookupBlock:
    """Tests for NoryTransactionLookupBlock."""

    def test_block_init(self):
        """Test block initialization."""
        block = NoryTransactionLookupBlock()
        assert block.id == "1334cf87-2b48-4e70-87d2-6e4807b78e02"

    @pytest.mark.asyncio
    async def test_run_success(self, mock_requests):
        """Test successful transaction lookup."""
        expected_response = {
            "transactionId": "abc123",
            "status": "confirmed",
            "confirmations": 10,
        }

        mock_requests_instance = MagicMock()
        mock_requests_instance.get = AsyncMock(
            return_value=MockResponse(expected_response)
        )
        mock_requests.return_value = mock_requests_instance

        block = NoryTransactionLookupBlock()
        input_data = block.Input(
            transaction_id="abc123",
            network=NoryNetwork.SOLANA_MAINNET,
            api_key="test_key",
        )

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "transaction"
        assert results[0][1]["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_url_encoding(self, mock_requests):
        """Test that transaction_id is URL-encoded."""
        mock_requests_instance = MagicMock()
        mock_requests_instance.get = AsyncMock(return_value=MockResponse({}))
        mock_requests.return_value = mock_requests_instance

        block = NoryTransactionLookupBlock()
        input_data = block.Input(
            transaction_id="tx/with/slashes",
            network=NoryNetwork.SOLANA_MAINNET,
        )

        async for _ in block.run(input_data):
            pass

        # Verify the URL was called with encoded transaction_id
        call_args = mock_requests_instance.get.call_args
        url = call_args[0][0]
        assert "tx%2Fwith%2Fslashes" in url
        assert "tx/with/slashes" not in url


class TestNoryHealthCheckBlock:
    """Tests for NoryHealthCheckBlock."""

    def test_block_init(self):
        """Test block initialization."""
        block = NoryHealthCheckBlock()
        assert block.id == "3b2595e1-5950-4094-b8b1-733dddd3b16c"

    @pytest.mark.asyncio
    async def test_run_success(self, mock_requests):
        """Test successful health check."""
        expected_response = {
            "status": "healthy",
            "supportedNetworks": ["solana-mainnet", "base-mainnet"],
        }

        mock_requests_instance = MagicMock()
        mock_requests_instance.get = AsyncMock(
            return_value=MockResponse(expected_response)
        )
        mock_requests.return_value = mock_requests_instance

        block = NoryHealthCheckBlock()
        input_data = block.Input()

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "health"
        assert results[0][1]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_run_error(self, mock_requests):
        """Test error handling."""
        mock_requests_instance = MagicMock()
        mock_requests_instance.get = AsyncMock(
            side_effect=Exception("Service unavailable")
        )
        mock_requests.return_value = mock_requests_instance

        block = NoryHealthCheckBlock()
        input_data = block.Input()

        results = []
        async for output_name, output_value in block.run(input_data):
            results.append((output_name, output_value))

        assert len(results) == 1
        assert results[0][0] == "error"
        assert "Service unavailable" in results[0][1]


class TestNoryNetwork:
    """Tests for NoryNetwork enum."""

    def test_network_values(self):
        """Test that all expected networks are defined."""
        assert NoryNetwork.SOLANA_MAINNET.value == "solana-mainnet"
        assert NoryNetwork.SOLANA_DEVNET.value == "solana-devnet"
        assert NoryNetwork.BASE_MAINNET.value == "base-mainnet"
        assert NoryNetwork.POLYGON_MAINNET.value == "polygon-mainnet"
        assert NoryNetwork.ARBITRUM_MAINNET.value == "arbitrum-mainnet"
        assert NoryNetwork.OPTIMISM_MAINNET.value == "optimism-mainnet"
        assert NoryNetwork.AVALANCHE_MAINNET.value == "avalanche-mainnet"
        assert NoryNetwork.SEI_MAINNET.value == "sei-mainnet"
        assert NoryNetwork.IOTEX_MAINNET.value == "iotex-mainnet"
