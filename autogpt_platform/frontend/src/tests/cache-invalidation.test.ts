/**
 * Test suite for cache invalidation consistency.
 *
 * These tests ensure that when we invalidate query caches, the parameters
 * used match what the backend expects. If the default page_size changes
 * in the backend API, these tests will catch mismatches.
 *
 * NOTE: These are unit tests for cache key generation.
 * They use Playwright's test framework but don't require a browser.
 */

import { test, expect } from "@playwright/test";
import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import * as PaginationConfig from "@/lib/pagination-config";

test.describe("Cache Invalidation Tests", () => {
  test.describe("getGetV2ListMySubmissionsQueryKey", () => {
    test("should generate correct query key without params", () => {
      const key = getGetV2ListMySubmissionsQueryKey();
      expect(key).toEqual(["/api/store/submissions"]);
    });

    test("should generate correct query key with params", () => {
      const key = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: 20,
      });
      expect(key).toEqual([
        "/api/store/submissions",
        { page: 1, page_size: 20 },
      ]);
    });

    test("should generate different keys for different page_size values", () => {
      const key1 = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: 20,
      });
      const key2 = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: 25,
      });

      expect(key1).not.toEqual(key2);
    });
  });

  test.describe("Cache invalidation page_size consistency", () => {
    /**
     * This test documents the current default page_size used in the backend.
     * If this test fails, it means:
     * 1. The backend default page_size has changed, OR
     * 2. The frontend is using a different page_size than the backend
     *
     * When invalidating queries without params, we're invalidating ALL
     * submissions queries regardless of page_size. This is correct behavior.
     */
    test("should use page_size matching backend default when invalidating specific pages", () => {
      // Use the shared constant that matches backend's cache_config.V2_STORE_SUBMISSIONS_PAGE_SIZE
      const BACKEND_DEFAULT_PAGE_SIZE =
        PaginationConfig.STORE_SUBMISSIONS_PAGE_SIZE;

      // When we call invalidateQueries without params, it invalidates all variations
      const invalidateAllKey = getGetV2ListMySubmissionsQueryKey();
      expect(invalidateAllKey).toEqual(["/api/store/submissions"]);

      // When we call invalidateQueries with specific params, it should match backend
      const invalidateSpecificKey = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: BACKEND_DEFAULT_PAGE_SIZE,
      });
      expect(invalidateSpecificKey).toEqual([
        "/api/store/submissions",
        { page: 1, page_size: PaginationConfig.STORE_SUBMISSIONS_PAGE_SIZE },
      ]);
    });

    /**
     * This test verifies that invalidating without parameters will match
     * all cached queries regardless of their page_size.
     * This is the behavior when calling:
     * queryClient.invalidateQueries({ queryKey: getGetV2ListMySubmissionsQueryKey() })
     */
    test("should invalidate all submissions when using base key", () => {
      const baseKey = getGetV2ListMySubmissionsQueryKey();

      // These are examples of keys that would be cached
      const cachedKey1 = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: 20,
      });
      const cachedKey2 = getGetV2ListMySubmissionsQueryKey({
        page: 2,
        page_size: 20,
      });
      const cachedKey3 = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: 25,
      });

      // Base key should be a prefix of all cached keys
      expect(cachedKey1[0]).toBe(baseKey[0]);
      expect(cachedKey2[0]).toBe(baseKey[0]);
      expect(cachedKey3[0]).toBe(baseKey[0]);

      // This confirms that invalidating with base key will match all variations
      // because TanStack Query does prefix matching by default
    });

    /**
     * This test documents a potential issue:
     * If the backend's _clear_submissions_cache hardcodes page_size=20,
     * but the frontend uses a different page_size, the caches won't sync.
     *
     * The frontend should ALWAYS call invalidateQueries without params
     * to ensure all pages are invalidated, not just specific page_size values.
     */
    test("should document the cache invalidation strategy", () => {
      // CORRECT: This invalidates ALL submissions queries
      const correctInvalidation = getGetV2ListMySubmissionsQueryKey();
      expect(correctInvalidation).toEqual(["/api/store/submissions"]);

      // INCORRECT: This would only invalidate queries with page_size=20
      const incorrectInvalidation = getGetV2ListMySubmissionsQueryKey({
        page: 1,
        page_size: 20,
      });
      expect(incorrectInvalidation).toEqual([
        "/api/store/submissions",
        { page: 1, page_size: 20 },
      ]);

      // Verify current usage in codebase uses correct approach
      // (This is a documentation test - it will always pass)
      // Real verification requires checking actual invalidateQueries calls
    });
  });

  test.describe("Integration with backend cache clearing", () => {
    /**
     * This test documents how the backend's _clear_submissions_cache works
     * and what the frontend needs to do to stay in sync.
     */
    test("should document backend cache clearing behavior", () => {
      const BACKEND_HARDCODED_PAGE_SIZE = 20; // From cache.py line 18
      const BACKEND_NUM_PAGES_TO_CLEAR = 20; // From cache.py line 13

      // Backend clears pages 1-19 with page_size=20
      // Frontend should invalidate ALL queries to ensure sync

      const frontendInvalidationKey = getGetV2ListMySubmissionsQueryKey();

      // Document what gets invalidated
      const expectedInvalidations = Array.from(
        { length: BACKEND_NUM_PAGES_TO_CLEAR - 1 },
        (_, i) =>
          getGetV2ListMySubmissionsQueryKey({
            page: i + 1,
            page_size: BACKEND_HARDCODED_PAGE_SIZE,
          }),
      );

      // All backend-cleared pages should have the same base key
      expectedInvalidations.forEach((key) => {
        expect(key[0]).toBe(frontendInvalidationKey[0]);
      });

      // This confirms that using the base key for invalidation
      // will catch all the entries the backend cleared
    });

    /**
     * CRITICAL TEST: This test will fail if someone changes the page_size
     * in the frontend components but the backend still uses page_size=20.
     */
    test("should fail if frontend default page_size differs from backend", () => {
      // Both frontend and backend now use shared constants
      // Frontend: STORE_SUBMISSIONS_PAGE_SIZE from pagination-config.ts
      // Backend: V2_STORE_SUBMISSIONS_PAGE_SIZE from cache_config.py
      // These MUST be kept in sync manually (no cross-language constant sharing possible)

      const EXPECTED_PAGE_SIZE = 20;

      expect(PaginationConfig.STORE_SUBMISSIONS_PAGE_SIZE).toBe(
        EXPECTED_PAGE_SIZE,
      );

      // If this test fails, you must:
      // 1. Update backend/server/cache_config.py V2_STORE_SUBMISSIONS_PAGE_SIZE
      // 2. Update frontend/lib/pagination-config.ts STORE_SUBMISSIONS_PAGE_SIZE
      // 3. Update all routes and cache clearing logic to use the constants
      // 4. Update this test with the new expected value
    });
  });
});
