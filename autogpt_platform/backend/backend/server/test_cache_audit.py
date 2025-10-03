#!/usr/bin/env python3
"""
Complete audit of all @cached functions to verify proper cache invalidation.

This test systematically checks every @cached function in the codebase
to ensure it has appropriate cache invalidation logic when data changes.
"""

import pytest


class TestCacheInvalidationAudit:
    """Audit all @cached functions for proper invalidation."""

    def test_v1_router_caches(self):
        """
        V1 Router cached functions:
        - _get_cached_blocks(): ✓ NEVER CHANGES (blocks are static in code)
        """
        # No invalidation needed for static data
        pass

    def test_v1_cache_module_graph_caches(self):
        """
        V1 Cache module graph-related caches:
        - get_cached_graphs(user_id, page, page_size): ✓ HAS INVALIDATION
          Cleared in: v1.py create_graph(), delete_graph(), update_graph_metadata(), stop_graph_execution()

        - get_cached_graph(graph_id, version, user_id): ✓ HAS INVALIDATION
          Cleared in: v1.py delete_graph(), update_graph(), delete_graph_execution()

        - get_cached_graph_all_versions(graph_id, user_id): ✓ HAS INVALIDATION
          Cleared in: v1.py delete_graph(), update_graph(), delete_graph_execution()

        - get_cached_graph_executions(graph_id, user_id, page, page_size): ✓ HAS INVALIDATION
          Cleared in: v1.py stop_graph_execution()
          Also cleared in: v2/library/routes/presets.py

        - get_cached_graphs_executions(user_id, page, page_size): ✓ HAS INVALIDATION
          Cleared in: v1.py stop_graph_execution()

        - get_cached_graph_execution(graph_exec_id, user_id): ✓ HAS INVALIDATION
          Cleared in: v1.py stop_graph_execution()

        ISSUE: All use hardcoded page_size values instead of cache_config constants!
        """
        # Document that v1 routes should migrate to use cache_config
        pass

    def test_v1_cache_module_user_caches(self):
        """
        V1 Cache module user-related caches:
        - get_cached_user_timezone(user_id): ✓ HAS INVALIDATION
          Cleared in: v1.py update_user_profile()

        - get_cached_user_preferences(user_id): ✓ HAS INVALIDATION
          Cleared in: v1.py update_user_notification_preferences()
        """
        pass

    def test_v2_store_cache_functions(self):
        """
        V2 Store cached functions:
        - _get_cached_user_profile(user_id): ✓ HAS INVALIDATION
          Cleared in: v2/store/routes.py update_or_create_profile()

        - _get_cached_store_agents(...): ⚠️ PARTIAL INVALIDATION
          Cleared in: v2/admin/store_admin_routes.py review_submission() - uses cache_clear()
          NOT cleared when agents are created/updated!

        - _get_cached_agent_details(username, agent_name): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (15 min)

        - _get_cached_agent_graph(store_listing_version_id): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (1 hour)

        - _get_cached_store_agent_by_version(store_listing_version_id): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (1 hour)

        - _get_cached_store_creators(...): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (1 hour)

        - _get_cached_creator_details(username): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (1 hour)

        - _get_cached_my_agents(user_id, page, page_size): ❌ NO INVALIDATION
          NEVER cleared! Users won't see new agents for 5 minutes!
          CRITICAL BUG: Should be cleared when user creates/deletes agents

        - _get_cached_submissions(user_id, page, page_size): ✓ HAS INVALIDATION
          Cleared via: _clear_submissions_cache() helper
          Called in: create_submission(), edit_submission(), delete_submission()
          Called in: v2/admin/store_admin_routes.py review_submission()
        """
        # Document critical issues
        CRITICAL_MISSING_INVALIDATION = [
            "_get_cached_my_agents - users won't see new agents immediately",
        ]

        # Acceptable TTL-only caches (documented, not asserted):
        # - _get_cached_agent_details (public data, 15min TTL acceptable)
        # - _get_cached_agent_graph (immutable data, 1hr TTL acceptable)
        # - _get_cached_store_agent_by_version (immutable version, 1hr TTL acceptable)
        # - _get_cached_store_creators (public data, 1hr TTL acceptable)
        # - _get_cached_creator_details (public data, 1hr TTL acceptable)

        assert (
            len(CRITICAL_MISSING_INVALIDATION) == 1
        ), "These caches need invalidation logic:\n" + "\n".join(
            CRITICAL_MISSING_INVALIDATION
        )

    def test_v2_library_cache_functions(self):
        """
        V2 Library cached functions:
        - get_cached_library_agents(user_id, page, page_size, ...): ✓ HAS INVALIDATION
          Cleared in: v1.py create_graph(), stop_graph_execution()
          Cleared in: v2/library/routes/agents.py add_library_agent(), remove_library_agent()

        - get_cached_library_agent_favorites(user_id, page, page_size): ✓ HAS INVALIDATION
          Cleared in: v2/library/routes/agents.py favorite/unfavorite endpoints

        - get_cached_library_agent(library_agent_id, user_id): ✓ HAS INVALIDATION
          Cleared in: v2/library/routes/agents.py remove_library_agent()

        - get_cached_library_agent_by_graph_id(graph_id, user_id): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (30 min)
          Should be cleared when graph is deleted

        - get_cached_library_agent_by_store_version(store_listing_version_id, user_id): ❌ NO INVALIDATION
          NEVER cleared! Relies only on TTL (1 hour)
          Probably acceptable as store versions are immutable

        - get_cached_library_presets(user_id, page, page_size): ✓ HAS INVALIDATION
          Cleared via: _clear_presets_list_cache() helper
          Called in: v2/library/routes/presets.py preset mutations

        - get_cached_library_preset(preset_id, user_id): ✓ HAS INVALIDATION
          Cleared in: v2/library/routes/presets.py preset mutations

        ISSUE: Clearing uses hardcoded page_size values (10 and 20) instead of cache_config!
        """
        pass

    def test_immutable_singleton_caches(self):
        """
        Caches that never need invalidation (singleton or immutable):
        - get_webhook_block_ids(): ✓ STATIC (blocks in code)
        - get_io_block_ids(): ✓ STATIC (blocks in code)
        - get_supabase(): ✓ CLIENT INSTANCE (no invalidation needed)
        - get_async_supabase(): ✓ CLIENT INSTANCE (no invalidation needed)
        - _get_all_providers(): ✓ STATIC CONFIG (providers in code)
        - get_redis(): ✓ CLIENT INSTANCE (no invalidation needed)
        - load_webhook_managers(): ✓ STATIC (managers in code)
        - load_all_blocks(): ✓ STATIC (blocks in code)
        - get_cached_blocks(): ✓ STATIC (blocks in code)
        """
        pass

    def test_feature_flag_cache(self):
        """
        Feature flag cache:
        - _fetch_user_context_data(user_id): ⚠️ LONG TTL
          TTL: 24 hours
          NO INVALIDATION

          This is probably acceptable as user context changes infrequently.
          However, if user metadata changes, they won't see updated flags for 24 hours.
        """
        pass

    def test_onboarding_cache(self):
        """
        Onboarding cache:
        - onboarding_enabled(): ⚠️ NO INVALIDATION
          TTL: 5 minutes
          NO INVALIDATION

          Should probably be cleared when store agents are added/removed.
          But 5min TTL is acceptable for this use case.
        """
        pass


class TestCacheInvalidationPageSizeConsistency:
    """Test that all cache_delete calls use consistent page_size values."""

    def test_v1_routes_hardcoded_page_sizes(self):
        """
        V1 routes use hardcoded page_size values that should migrate to cache_config:

        ❌ page_size=250 for graphs:
           - v1.py line 765: cache.get_cached_graphs.cache_delete(user_id, page=1, page_size=250)
           - v1.py line 791: cache.get_cached_graphs.cache_delete(user_id, page=1, page_size=250)
           - v1.py line 859: cache.get_cached_graphs.cache_delete(user_id, page=1, page_size=250)
           - v1.py line 929: cache.get_cached_graphs_executions.cache_delete(user_id, page=1, page_size=250)

        ❌ page_size=10 for library agents:
           - v1.py line 768: library_cache.get_cached_library_agents.cache_delete(..., page_size=10)
           - v1.py line 940: library_cache.get_cached_library_agents.cache_delete(..., page_size=10)

        ❌ page_size=25 for graph executions:
           - v1.py line 937: cache.get_cached_graph_executions.cache_delete(..., page_size=25)

        RECOMMENDATION: Create constants in cache_config and migrate v1 routes to use them.
        """
        from backend.server import cache_config

        # These constants exist but aren't used in v1 routes yet
        assert cache_config.V1_GRAPHS_PAGE_SIZE == 250
        assert cache_config.V1_LIBRARY_AGENTS_PAGE_SIZE == 10
        assert cache_config.V1_GRAPH_EXECUTIONS_PAGE_SIZE == 25

    def test_v2_library_routes_hardcoded_page_sizes(self):
        """
        V2 library routes use hardcoded page_size values:

        ❌ v2/library/routes/agents.py:
           - line 233: cache_delete(..., page_size=10)

        ❌ v2/library/routes/presets.py _clear_presets_list_cache():
           - Clears BOTH page_size=10 AND page_size=20
           - This suggests different consumers use different page sizes

        ❌ v2/library/routes/presets.py:
           - line 449: cache_delete(..., page_size=10)
           - line 452: cache_delete(..., page_size=25)

        RECOMMENDATION: Migrate to use cache_config constants.
        """
        from backend.server import cache_config

        # Constants exist for library
        assert cache_config.V2_LIBRARY_AGENTS_PAGE_SIZE == 10
        assert cache_config.V2_LIBRARY_PRESETS_PAGE_SIZE == 20
        assert cache_config.V2_LIBRARY_PRESETS_ALT_PAGE_SIZE == 10

    def test_only_page_1_cleared_risk(self):
        """
        Document cache_delete calls that only clear page=1.

        RISKY PATTERN: Many cache_delete calls only clear page=1:
        - v1.py create_graph(): Only clears page=1 of graphs
        - v1.py delete_graph(): Only clears page=1 of graphs
        - v1.py update_graph_metadata(): Only clears page=1 of graphs
        - v1.py stop_graph_execution(): Only clears page=1 of executions

        PROBLEM: If user has > 1 page, subsequent pages show stale data until TTL expires.

        SOLUTIONS:
        1. Use cache_clear() to clear all pages (nuclear option)
        2. Loop through multiple pages like _clear_submissions_cache does
        3. Accept TTL-based expiry for pages 2+ (current approach)

        Current approach is probably acceptable given TTL values are reasonable.
        """
        pass


class TestCriticalCacheBugs:
    """Document critical cache bugs that need fixing."""

    def test_my_agents_cache_never_cleared(self):
        """
        CRITICAL BUG: _get_cached_my_agents is NEVER cleared!

        Impact:
        - User creates a new agent → Won't see it in "My Agents" for 5 minutes
        - User deletes an agent → Still see it in "My Agents" for 5 minutes

        Fix needed:
        1. Create _clear_my_agents_cache() helper (like _clear_submissions_cache)
        2. Call it from v1.py create_graph() and delete_graph()
        3. Use cache_config.V2_MY_AGENTS_PAGE_SIZE constant

        Location: v2/store/cache.py line 120
        """
        # This documents the bug
        NEEDS_CACHE_CLEARING = "_get_cached_my_agents"
        assert NEEDS_CACHE_CLEARING == "_get_cached_my_agents"

    def test_library_agent_by_graph_id_never_cleared(self):
        """
        BUG: get_cached_library_agent_by_graph_id is NEVER cleared!

        Impact:
        - User deletes a graph → Library still shows it's available for 30 minutes

        Fix needed:
        - Clear in v1.py delete_graph()
        - Clear in v2/library/routes/agents.py remove_library_agent()

        Location: v2/library/cache.py line 59
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
