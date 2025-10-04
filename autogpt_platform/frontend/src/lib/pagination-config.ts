/**
 * Shared pagination configuration constants.
 *
 * These values MUST match the backend's cache_config.py to ensure
 * proper cache invalidation when data is mutated.
 *
 * CRITICAL: If you change any of these values:
 * 1. Update backend/server/cache_config.py
 * 2. Update cache invalidation logic
 * 3. Run tests to ensure consistency
 */

/**
 * Default page size for store agents listing
 * Backend: V2_STORE_AGENTS_PAGE_SIZE
 */
export const STORE_AGENTS_PAGE_SIZE = 20;

/**
 * Default page size for store creators listing
 * Backend: V2_STORE_CREATORS_PAGE_SIZE
 */
export const STORE_CREATORS_PAGE_SIZE = 20;

/**
 * Default page size for user submissions listing
 * Backend: V2_STORE_SUBMISSIONS_PAGE_SIZE
 */
export const STORE_SUBMISSIONS_PAGE_SIZE = 20;

/**
 * Default page size for user's own agents listing
 * Backend: V2_MY_AGENTS_PAGE_SIZE
 */
export const MY_AGENTS_PAGE_SIZE = 20;

/**
 * Default page size for library agents listing
 * Backend: V2_LIBRARY_AGENTS_PAGE_SIZE
 */
export const LIBRARY_AGENTS_PAGE_SIZE = 10;

/**
 * Default page size for library presets listing
 * Backend: V2_LIBRARY_PRESETS_PAGE_SIZE
 */
export const LIBRARY_PRESETS_PAGE_SIZE = 20;

/**
 * Default page size for agent runs/executions
 * Backend: V1_GRAPH_EXECUTIONS_PAGE_SIZE (note: this is from v1 API)
 */
export const AGENT_RUNS_PAGE_SIZE = 20;

/**
 * Large page size for fetching "all" items (marketplace top agents)
 * Used when we want to fetch a comprehensive list without pagination UI
 */
export const LARGE_PAGE_SIZE = 1000;

/**
 * Very large page size for specific use cases
 * Used in agent runs view for comprehensive listing
 */
export const EXTRA_LARGE_PAGE_SIZE = 100;
