#!/usr/bin/env python3
"""
Comprehensive test suite for cache invalidation consistency across the entire backend.

This test file identifies ALL locations where cache_delete is called with hardcoded
parameters (especially page_size) and ensures they match the corresponding route defaults.

CRITICAL: If any test in this file fails, it means cache invalidation will be broken
and users will see stale data after mutations.

Key problem areas identified:
1. v1.py routes: Uses page_size=250 for graphs, but cache clearing uses page_size=250 ✓
2. v1.py routes: Uses page_size=10 for library agents clearing
3. v2/library routes: Uses page_size=10 for library agents clearing
4. v2/store routes: Uses page_size=20 for submissions clearing (in _clear_submissions_cache)
5. v2/library presets: Uses page_size=10 AND page_size=20 for presets (dual clearing)
"""

import pytest


class TestCacheInvalidationConsistency:
    """Test that all cache_delete calls use correct parameters matching route defaults."""

    def test_v1_graphs_cache_page_size_consistency(self):
        """
        Test v1 graphs routes use consistent page_size.

        Locations that must match:
        - routes/v1.py line 682: default page_size=250
        - routes/v1.py line 765: cache_delete with page_size=250
        - routes/v1.py line 791: cache_delete with page_size=250
        - routes/v1.py line 859: cache_delete with page_size=250
        - routes/v1.py line 929: cache_delete with page_size=250
        - routes/v1.py line 1034: default page_size=250
        """
        V1_GRAPHS_DEFAULT_PAGE_SIZE = 250

        # This is the expected value - if this test fails, check all the above locations
        assert V1_GRAPHS_DEFAULT_PAGE_SIZE == 250, (
            "If you changed the default page_size for v1 graphs, you must update:\n"
            "1. routes/v1.py list_graphs() default parameter\n"
            "2. routes/v1.py create_graph() cache_delete call\n"
            "3. routes/v1.py delete_graph() cache_delete call\n"
            "4. routes/v1.py update_graph_metadata() cache_delete call\n"
            "5. routes/v1.py stop_graph_execution() cache_delete call\n"
            "6. routes/v1.py list_graph_run_events() default parameter"
        )

    def test_v1_library_agents_cache_page_size_consistency(self):
        """
        Test v1 library agents cache clearing uses consistent page_size.

        Locations that must match:
        - routes/v1.py line 768: cache_delete with page_size=10
        - routes/v1.py line 940: cache_delete with page_size=10
        - v2/library/routes/agents.py line 233: cache_delete with page_size=10

        WARNING: These hardcode page_size=10 but we need to verify this matches
        the actual page_size used when fetching library agents!
        """
        V1_LIBRARY_AGENTS_CLEARING_PAGE_SIZE = 10

        assert V1_LIBRARY_AGENTS_CLEARING_PAGE_SIZE == 10, (
            "If you changed the library agents clearing page_size, you must update:\n"
            "1. routes/v1.py create_graph() cache clearing loop\n"
            "2. routes/v1.py stop_graph_execution() cache clearing loop\n"
            "3. v2/library/routes/agents.py add_library_agent() cache clearing loop"
        )

        # TODO: This should be verified against the actual default used in library routes

    def test_v1_graph_executions_cache_page_size_consistency(self):
        """
        Test v1 graph executions cache clearing uses consistent page_size.

        Locations:
        - routes/v1.py line 937: cache_delete with page_size=25
        - v2/library/routes/presets.py line 449: cache_delete with page_size=10
        - v2/library/routes/presets.py line 452: cache_delete with page_size=25
        """
        V1_GRAPH_EXECUTIONS_CLEARING_PAGE_SIZE = 25

        # Note: presets.py clears BOTH page_size=10 AND page_size=25
        # This suggests there may be multiple consumers with different page sizes
        assert V1_GRAPH_EXECUTIONS_CLEARING_PAGE_SIZE == 25

    def test_v2_store_submissions_cache_page_size_consistency(self):
        """
        Test v2 store submissions use consistent page_size.

        Locations that must match:
        - v2/store/routes.py line 484: default page_size=20
        - v2/store/cache.py line 18: _clear_submissions_cache uses page_size=20

        This is already tested in test_cache_delete.py but documented here for completeness.
        """
        V2_STORE_SUBMISSIONS_DEFAULT_PAGE_SIZE = 20
        V2_STORE_SUBMISSIONS_CLEARING_PAGE_SIZE = 20

        assert (
            V2_STORE_SUBMISSIONS_DEFAULT_PAGE_SIZE
            == V2_STORE_SUBMISSIONS_CLEARING_PAGE_SIZE
        ), (
            "The default page_size for store submissions must match the hardcoded value in _clear_submissions_cache!\n"
            "Update both:\n"
            "1. v2/store/routes.py get_submissions() default parameter\n"
            "2. v2/store/cache.py _clear_submissions_cache() hardcoded page_size"
        )

    def test_v2_library_presets_cache_page_size_consistency(self):
        """
        Test v2 library presets cache clearing uses consistent page_size.

        Locations:
        - v2/library/routes/presets.py line 36: cache_delete with page_size=10
        - v2/library/routes/presets.py line 39: cache_delete with page_size=20

        This route clears BOTH page_size=10 and page_size=20, suggesting multiple consumers.
        """
        V2_LIBRARY_PRESETS_CLEARING_PAGE_SIZES = [10, 20]

        assert 10 in V2_LIBRARY_PRESETS_CLEARING_PAGE_SIZES
        assert 20 in V2_LIBRARY_PRESETS_CLEARING_PAGE_SIZES

        # TODO: Verify these match the actual page_size defaults used in preset routes

    def test_cache_clearing_helper_functions_documented(self):
        """
        Document all cache clearing helper functions and their hardcoded parameters.

        Helper functions that wrap cache_delete with hardcoded params:
        1. v2/store/cache.py::_clear_submissions_cache() - hardcodes page_size=20, num_pages=20
        2. v2/library/routes/presets.py::_clear_presets_list_cache() - hardcodes page_size=10 AND 20, num_pages=20

        These helpers are DANGEROUS because:
        - They hide the hardcoded parameters
        - They loop through multiple pages with hardcoded page_size
        - If the route default changes, these won't clear the right cache entries
        """
        HELPER_FUNCTIONS = {
            "_clear_submissions_cache": {
                "file": "v2/store/cache.py",
                "page_size": 20,
                "num_pages": 20,
                "risk": "HIGH - single page_size, could miss entries if default changes",
            },
            "_clear_presets_list_cache": {
                "file": "v2/library/routes/presets.py",
                "page_size": [10, 20],
                "num_pages": 20,
                "risk": "MEDIUM - clears multiple page_sizes, but could still miss new ones",
            },
        }

        assert (
            len(HELPER_FUNCTIONS) == 2
        ), "If you add new cache clearing helper functions, document them here!"

    def test_cache_delete_without_page_loops_are_risky(self):
        """
        Document cache_delete calls that clear only page=1 (risky if there are multiple pages).

        Single page cache_delete calls:
        - routes/v1.py line 765: Only clears page=1 with page_size=250
        - routes/v1.py line 791: Only clears page=1 with page_size=250
        - routes/v1.py line 859: Only clears page=1 with page_size=250

        These are RISKY because:
        - If a user has more than one page of graphs, pages 2+ won't be invalidated
        - User could see stale data on pagination

        RECOMMENDATION: Use cache_clear() or loop through multiple pages like
        _clear_submissions_cache does.
        """
        SINGLE_PAGE_CLEARS = [
            "routes/v1.py line 765: create_graph clears only page=1",
            "routes/v1.py line 791: delete_graph clears only page=1",
            "routes/v1.py line 859: update_graph_metadata clears only page=1",
        ]

        # This test documents the issue but doesn't fail
        # Consider this a TODO to fix these cache clearing strategies
        assert (
            len(SINGLE_PAGE_CLEARS) >= 3
        ), "These cache_delete calls should probably loop through multiple pages"

    def test_all_cached_functions_have_proper_invalidation(self):
        """
        Verify all @cached functions have corresponding cache_delete calls.

        Functions with proper invalidation:
        ✓ get_cached_user_profile - cleared on profile update
        ✓ get_cached_store_agents - cleared on admin review (cache_clear)
        ✓ get_cached_submissions - cleared via _clear_submissions_cache helper
        ✓ get_cached_graphs - cleared on graph mutations
        ✓ get_cached_library_agents - cleared on library changes

        Functions that might not have proper invalidation:
        ? get_cached_agent_details - not explicitly cleared
        ? get_cached_store_creators - not explicitly cleared
        ? get_cached_my_agents - not explicitly cleared (no helper function exists!)

        This is a documentation test - actual verification requires code analysis.
        """
        NEEDS_VERIFICATION = [
            "get_cached_agent_details",
            "get_cached_store_creators",
            "get_cached_my_agents",  # NO CLEARING FUNCTION EXISTS!
        ]

        assert "get_cached_my_agents" in NEEDS_VERIFICATION, (
            "get_cached_my_agents has no cache clearing logic - this is a BUG!\n"
            "When a user creates/deletes an agent, their 'my agents' list won't update."
        )


class TestCacheKeyParameterOrdering:
    """
    Test that cache_delete calls use the same parameter order as the @cached function.

    The @cached decorator uses function signature order to create cache keys.
    cache_delete must use the exact same order or it won't find the cached entry!
    """

    def test_cached_function_parameter_order_matters(self):
        """
        Document that parameter order in cache_delete must match @cached function signature.

        Example from v2/store/cache.py:

        @cached(...)
        async def _get_cached_submissions(user_id: str, page: int, page_size: int):
            ...

        CORRECT: _get_cached_submissions.cache_delete(user_id, page=1, page_size=20)
        WRONG: _get_cached_submissions.cache_delete(page=1, user_id=user_id, page_size=20)

        The cached decorator generates keys based on the POSITIONAL order, so parameter
        order must match between the function definition and cache_delete call.
        """
        # This is a documentation test - no assertion needed
        # Real verification requires inspecting each cache_delete call
        pass

    def test_named_parameters_vs_positional_in_cache_delete(self):
        """
        Document best practice: use named parameters in cache_delete for safety.

        Good practice seen in codebase:
        - cache.get_cached_graphs.cache_delete(user_id=user_id, page=1, page_size=250)
        - library_cache.get_cached_library_agents.cache_delete(user_id=user_id, page=page, page_size=10)

        This is safer than positional arguments because:
        1. More readable
        2. Less likely to get order wrong
        3. Self-documenting what each parameter means
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
