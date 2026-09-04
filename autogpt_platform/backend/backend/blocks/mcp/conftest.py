import pytest

from backend.blocks.mcp.protocol import era_cache


@pytest.fixture(autouse=True)
def _fresh_era_cache():
    """Era detection is cached per process; keep tests independent of order."""
    era_cache.clear()
    yield
    era_cache.clear()
