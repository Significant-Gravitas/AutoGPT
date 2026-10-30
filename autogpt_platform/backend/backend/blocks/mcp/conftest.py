from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.protocol import era_cache


@pytest.fixture(autouse=True)
def _fresh_era_cache():
    """Era detection is cached per process; keep tests independent of order."""
    era_cache.clear()
    yield
    era_cache.clear()


@pytest.fixture
def no_stored_mcp_credentials():
    with patch.object(
        MCPToolBlock, "_auto_lookup_credential", AsyncMock(return_value=None)
    ):
        yield
