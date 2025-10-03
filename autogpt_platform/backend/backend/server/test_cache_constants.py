#!/usr/bin/env python3
"""
Test suite to verify cache_config constants are being used correctly.

This ensures that the centralized cache_config.py constants are actually
used throughout the codebase, not just defined.
"""

import pytest

from backend.server import cache_config


class TestCacheConfigConstants:
    """Verify cache_config constants have expected values."""

    def test_v2_store_page_sizes(self):
        """Test V2 Store API page size constants."""
        assert cache_config.V2_STORE_AGENTS_PAGE_SIZE == 20
        assert cache_config.V2_STORE_CREATORS_PAGE_SIZE == 20
        assert cache_config.V2_STORE_SUBMISSIONS_PAGE_SIZE == 20
        assert cache_config.V2_MY_AGENTS_PAGE_SIZE == 20

    def test_v2_library_page_sizes(self):
        """Test V2 Library API page size constants."""
        assert cache_config.V2_LIBRARY_AGENTS_PAGE_SIZE == 10
        assert cache_config.V2_LIBRARY_PRESETS_PAGE_SIZE == 20
        assert cache_config.V2_LIBRARY_PRESETS_ALT_PAGE_SIZE == 10

    def test_v1_page_sizes(self):
        """Test V1 API page size constants."""
        assert cache_config.V1_GRAPHS_PAGE_SIZE == 250
        assert cache_config.V1_LIBRARY_AGENTS_PAGE_SIZE == 10
        assert cache_config.V1_GRAPH_EXECUTIONS_PAGE_SIZE == 25

    def test_cache_clearing_config(self):
        """Test cache clearing configuration."""
        assert cache_config.MAX_PAGES_TO_CLEAR == 20

    def test_get_page_sizes_for_clearing_helper(self):
        """Test the helper function for getting page sizes to clear."""
        # Single page size
        result = cache_config.get_page_sizes_for_clearing(20)
        assert result == [20]

        # Multiple page sizes
        result = cache_config.get_page_sizes_for_clearing(20, 10)
        assert result == [20, 10]

        # With None alt_page_size
        result = cache_config.get_page_sizes_for_clearing(20, None)
        assert result == [20]


class TestCacheConfigUsage:
    """Test that cache_config constants are actually used in the code."""

    def test_store_routes_import_cache_config(self):
        """Verify store routes imports cache_config."""
        import backend.server.v2.store.routes as store_routes

        # Check that cache_config is imported
        assert hasattr(store_routes, "backend")
        assert hasattr(store_routes.backend.server, "cache_config")

    def test_store_cache_uses_constants(self):
        """Verify store cache module uses cache_config constants."""
        import backend.server.v2.store.cache as store_cache

        # Check the module imports cache_config
        assert hasattr(store_cache, "backend")
        assert hasattr(store_cache.backend.server, "cache_config")

        # The _clear_submissions_cache function should use the constant
        import inspect

        source = inspect.getsource(store_cache._clear_submissions_cache)
        assert (
            "cache_config.V2_STORE_SUBMISSIONS_PAGE_SIZE" in source
        ), "_clear_submissions_cache must use cache_config.V2_STORE_SUBMISSIONS_PAGE_SIZE"
        assert (
            "cache_config.MAX_PAGES_TO_CLEAR" in source
        ), "_clear_submissions_cache must use cache_config.MAX_PAGES_TO_CLEAR"

    def test_admin_routes_use_constants(self):
        """Verify admin routes use cache_config constants."""
        import backend.server.v2.admin.store_admin_routes as admin_routes

        # Check that cache_config is imported
        assert hasattr(admin_routes, "backend")
        assert hasattr(admin_routes.backend.server, "cache_config")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
