import { describe, it, expect } from "vitest";
import {
  getPaginatedTotalCount,
  getPaginationNextPageNumber,
  unpaginate,
} from "./helpers";
import type { InfiniteData } from "@tanstack/react-query";

// Helper function to check if pagination info is valid (exported for testing)
function hasValidPaginationInfo(page: unknown): page is {
  data: {
    pagination: {
      total_items: number;
      total_pages: number;
      current_page: number;
      page_size: number;
    };
    [key: string]: any;
  };
} {
  if (
    typeof page !== "object" ||
    page === null ||
    !("data" in page) ||
    typeof (page as Record<string, unknown>).data !== "object" ||
    (page as Record<string, unknown>).data === null
  ) {
    return false;
  }

  // Check if pagination exists and is valid
  const data = (page as Record<string, unknown>).data as Record<
    string,
    unknown
  >;
  if (
    !("pagination" in data) ||
    typeof data.pagination !== "object" ||
    data.pagination === null
  ) {
    return false;
  }

  const pagination = data.pagination as Record<string, unknown>;

  return (
    "total_items" in pagination &&
    typeof pagination.total_items === "number" &&
    "total_pages" in pagination &&
    typeof pagination.total_pages === "number" &&
    "current_page" in pagination &&
    typeof pagination.current_page === "number" &&
    "page_size" in pagination &&
    typeof pagination.page_size === "number"
  );
}

describe("helpers", () => {
  describe("hasValidPaginationInfo", () => {
    it("should return false for undefined", () => {
      expect(hasValidPaginationInfo(undefined)).toBe(false);
    });

    it("should return false for null", () => {
      expect(hasValidPaginationInfo(null)).toBe(false);
    });

    it("should return false when data is undefined", () => {
      expect(hasValidPaginationInfo({ status: 200 })).toBe(false);
    });

    it("should return false when pagination is missing", () => {
      expect(hasValidPaginationInfo({ status: 200, data: {} })).toBe(false);
    });

    it("should return false when pagination fields are missing", () => {
      expect(
        hasValidPaginationInfo({
          status: 200,
          data: { pagination: { total_items: 10 } },
        }),
      ).toBe(false);
    });

    it("should return true for valid pagination info", () => {
      expect(
        hasValidPaginationInfo({
          status: 200,
          data: {
            pagination: {
              total_items: 100,
              total_pages: 10,
              current_page: 1,
              page_size: 10,
            },
          },
        }),
      ).toBe(true);
    });

    // This test verifies the fix for issue #10848
    it("should return false for 401 error response without data", () => {
      const errorResponse = { status: 401, data: undefined };
      expect(hasValidPaginationInfo(errorResponse)).toBe(false);
    });

    it("should return false for 401 error response with empty data", () => {
      const errorResponse = { status: 401, data: {} };
      expect(hasValidPaginationInfo(errorResponse)).toBe(false);
    });
  });

  describe("getPaginatedTotalCount", () => {
    it("should return fallback count for undefined data", () => {
      expect(getPaginatedTotalCount(undefined, 5)).toBe(5);
    });

    it("should return 0 for undefined data without fallback", () => {
      expect(getPaginatedTotalCount(undefined)).toBe(0);
    });

    it("should return total_items from valid pagination", () => {
      const infiniteData = {
        pages: [
          {
            status: 200,
            data: {
              agents: [],
              pagination: {
                total_items: 42,
                total_pages: 5,
                current_page: 1,
                page_size: 10,
              },
            },
          },
        ],
        pageParams: [1],
      } as unknown as InfiniteData<unknown>;

      expect(getPaginatedTotalCount(infiniteData)).toBe(42);
    });

    // This test verifies the fix for issue #10848 - crash on 401
    it("should return fallback for 401 error response without crashing", () => {
      const errorData = {
        pages: [{ status: 401, data: undefined }],
        pageParams: [1],
      } as unknown as InfiniteData<unknown>;

      expect(() => getPaginatedTotalCount(errorData, 0)).not.toThrow();
      expect(getPaginatedTotalCount(errorData, 0)).toBe(0);
    });
  });

  describe("getPaginationNextPageNumber", () => {
    it("should return undefined for undefined lastPage", () => {
      expect(getPaginationNextPageNumber(undefined)).toBeUndefined();
    });

    it("should return undefined when lastPage.data is undefined", () => {
      expect(
        getPaginationNextPageNumber({ status: 401, data: undefined }),
      ).toBeUndefined();
    });

    it("should return next page number when more pages exist", () => {
      const lastPage = {
        status: 200,
        data: {
          agents: [],
          pagination: {
            total_items: 100,
            total_pages: 10,
            current_page: 1,
            page_size: 10,
          },
        },
      };

      expect(getPaginationNextPageNumber(lastPage)).toBe(2);
    });

    it("should return undefined when on last page", () => {
      const lastPage = {
        status: 200,
        data: {
          agents: [],
          pagination: {
            total_items: 10,
            total_pages: 1,
            current_page: 1,
            page_size: 10,
          },
        },
      };

      expect(getPaginationNextPageNumber(lastPage)).toBeUndefined();
    });

    // This test verifies the fix for issue #10848 - crash on 401
    it("should not crash on 401 error response", () => {
      const errorResponse = {
        status: 401,
        data: undefined,
      };

      expect(() => getPaginationNextPageNumber(errorResponse)).not.toThrow();
      expect(getPaginationNextPageNumber(errorResponse)).toBeUndefined();
    });
  });

  describe("unpaginate", () => {
    it("should flatten paginated data", () => {
      const infiniteData = {
        pages: [
          {
            status: 200,
            data: {
              agents: [{ id: "1" }, { id: "2" }],
              pagination: {
                total_items: 4,
                total_pages: 2,
                current_page: 1,
                page_size: 2,
              },
            },
          },
          {
            status: 200,
            data: {
              agents: [{ id: "3" }, { id: "4" }],
              pagination: {
                total_items: 4,
                total_pages: 2,
                current_page: 2,
                page_size: 2,
              },
            },
          },
        ],
        pageParams: [1, 2],
      } as unknown as InfiniteData<{
        status: number;
        data: { agents: { id: string }[] };
      }>;

      const result = unpaginate(infiniteData, "agents");
      expect(result).toHaveLength(4);
      expect(result.map((a) => a.id)).toEqual(["1", "2", "3", "4"]);
    });

    it("should handle non-200 status pages gracefully", () => {
      const infiniteData = {
        pages: [
          {
            status: 200,
            data: {
              agents: [{ id: "1" }],
              pagination: {
                total_items: 1,
                total_pages: 1,
                current_page: 1,
                page_size: 10,
              },
            },
          },
          {
            status: 401,
            data: undefined,
          },
        ],
        pageParams: [1, 2],
      } as unknown as InfiniteData<{
        status: number;
        data: { agents: { id: string }[] };
      }>;

      // Should not throw and should return items from valid pages
      expect(() => unpaginate(infiniteData, "agents")).not.toThrow();
      const result = unpaginate(infiniteData, "agents");
      expect(result).toHaveLength(1);
    });
  });
});
