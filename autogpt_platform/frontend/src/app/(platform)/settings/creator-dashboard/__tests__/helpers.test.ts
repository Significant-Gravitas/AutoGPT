import { describe, expect, test } from "vitest";

import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";

import {
  applyFiltersAndSort,
  computeStats,
  filterSubmissions,
  formatRuns,
  formatSubmittedAt,
  getStatusVisual,
  INITIAL_FILTER_STATE,
  isFiltered,
  STATUS_FILTERS,
  STATUS_OPTIONS,
  STATUS_VISUAL,
  type FilterState,
} from "../helpers";

function makeSubmission(
  overrides: Partial<StoreSubmission> = {},
): StoreSubmission {
  return {
    listing_id: "listing-1",
    user_id: "user-1",
    slug: "agent-slug",
    listing_version_id: "lv-1",
    listing_version: 1,
    graph_id: "graph-1",
    graph_version: 1,
    name: "Test Agent",
    sub_heading: "sub",
    description: "desc",
    instructions: null,
    categories: [],
    image_urls: [],
    video_url: null,
    agent_output_demo_url: null,
    submitted_at: null,
    changes_summary: null,
    status: SubmissionStatus.PENDING,
    run_count: 0,
    review_count: 0,
    review_avg_rating: 0,
    ...overrides,
  };
}

describe("creator-dashboard helpers", () => {
  describe("INITIAL_FILTER_STATE", () => {
    test("starts with no filters and descending sort direction", () => {
      expect(INITIAL_FILTER_STATE).toEqual({
        statuses: [],
        nameQuery: "",
        sortKey: null,
        sortDir: "desc",
      });
    });
  });

  describe("isFiltered", () => {
    test("returns false on initial state", () => {
      expect(isFiltered(INITIAL_FILTER_STATE)).toBe(false);
    });

    test("returns true if any status selected", () => {
      const state: FilterState = {
        ...INITIAL_FILTER_STATE,
        statuses: [SubmissionStatus.APPROVED],
      };
      expect(isFiltered(state)).toBe(true);
    });

    test("returns true if name query is non-empty after trim", () => {
      expect(isFiltered({ ...INITIAL_FILTER_STATE, nameQuery: "foo" })).toBe(
        true,
      );
      expect(isFiltered({ ...INITIAL_FILTER_STATE, nameQuery: "   " })).toBe(
        false,
      );
    });

    test("returns true if sortKey is set", () => {
      expect(isFiltered({ ...INITIAL_FILTER_STATE, sortKey: "runs" })).toBe(
        true,
      );
    });
  });

  describe("applyFiltersAndSort", () => {
    const items = [
      makeSubmission({
        listing_version_id: "a",
        name: "Alpha",
        status: SubmissionStatus.APPROVED,
        run_count: 100,
        review_avg_rating: 4.5,
        submitted_at: new Date("2026-01-10"),
      }),
      makeSubmission({
        listing_version_id: "b",
        name: "Beta Agent",
        status: SubmissionStatus.PENDING,
        run_count: 50,
        review_avg_rating: 3,
        submitted_at: new Date("2026-02-10"),
      }),
      makeSubmission({
        listing_version_id: "c",
        name: "Gamma",
        status: SubmissionStatus.REJECTED,
        run_count: 999,
        review_avg_rating: 5,
        submitted_at: new Date("2026-03-10"),
      }),
    ];

    test("returns same list when no filters or sort applied", () => {
      const result = applyFiltersAndSort(items, INITIAL_FILTER_STATE);
      expect(result).toHaveLength(3);
      expect(result.map((s) => s.listing_version_id)).toEqual(["a", "b", "c"]);
    });

    test("filters by status set", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        statuses: [SubmissionStatus.APPROVED, SubmissionStatus.REJECTED],
      });
      expect(result.map((s) => s.listing_version_id)).toEqual(["a", "c"]);
    });

    test("filters by case-insensitive name query", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        nameQuery: "BETA",
      });
      expect(result).toHaveLength(1);
      expect(result[0].listing_version_id).toBe("b");
    });

    test("ignores whitespace-only name query", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        nameQuery: "   ",
      });
      expect(result).toHaveLength(3);
    });

    test("sorts by runs descending", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        sortKey: "runs",
        sortDir: "desc",
      });
      expect(result.map((s) => s.listing_version_id)).toEqual(["c", "a", "b"]);
    });

    test("sorts by runs ascending", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        sortKey: "runs",
        sortDir: "asc",
      });
      expect(result.map((s) => s.listing_version_id)).toEqual(["b", "a", "c"]);
    });

    test("sorts by rating descending", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        sortKey: "rating",
        sortDir: "desc",
      });
      expect(result.map((s) => s.listing_version_id)).toEqual(["c", "a", "b"]);
    });

    test("sorts by submitted date descending (newest first)", () => {
      const result = applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        sortKey: "submitted",
        sortDir: "desc",
      });
      expect(result.map((s) => s.listing_version_id)).toEqual(["c", "b", "a"]);
    });

    test("treats invalid submitted_at strings as 0 (NaN guard)", () => {
      const withInvalid = [
        ...items,
        makeSubmission({
          listing_version_id: "bad",
          submitted_at: "not-a-date" as unknown as Date,
        }),
      ];
      const result = applyFiltersAndSort(withInvalid, {
        ...INITIAL_FILTER_STATE,
        sortKey: "submitted",
        sortDir: "asc",
      });
      expect(result[0].listing_version_id).toBe("bad");
    });

    test("treats null submitted_at as 0 when sorting", () => {
      const withNull = [
        ...items,
        makeSubmission({
          listing_version_id: "d",
          submitted_at: null,
        }),
      ];
      const result = applyFiltersAndSort(withNull, {
        ...INITIAL_FILTER_STATE,
        sortKey: "submitted",
        sortDir: "asc",
      });
      expect(result[0].listing_version_id).toBe("d");
    });

    test("does not mutate original array when sorting", () => {
      const original = [...items];
      applyFiltersAndSort(items, {
        ...INITIAL_FILTER_STATE,
        sortKey: "runs",
        sortDir: "asc",
      });
      expect(items.map((s) => s.listing_version_id)).toEqual(
        original.map((s) => s.listing_version_id),
      );
    });
  });

  describe("computeStats", () => {
    test("returns zeroed stats with null average for empty list", () => {
      expect(computeStats([])).toEqual({
        total: 0,
        approved: 0,
        pending: 0,
        totalRuns: 0,
        averageRating: null,
      });
    });

    test("aggregates totals, approved, pending, runs, and average rating", () => {
      const list = [
        makeSubmission({
          status: SubmissionStatus.APPROVED,
          run_count: 100,
          review_avg_rating: 4,
        }),
        makeSubmission({
          status: SubmissionStatus.APPROVED,
          run_count: 200,
          review_avg_rating: 5,
        }),
        makeSubmission({
          status: SubmissionStatus.PENDING,
          run_count: 50,
          review_avg_rating: 0,
        }),
        makeSubmission({
          status: SubmissionStatus.REJECTED,
          run_count: 10,
        }),
      ];

      expect(computeStats(list)).toEqual({
        total: 4,
        approved: 2,
        pending: 1,
        totalRuns: 360,
        averageRating: 4.5,
      });
    });

    test("returns null average when no submission has a positive rating", () => {
      const list = [
        makeSubmission({ review_avg_rating: 0 }),
        makeSubmission({ review_avg_rating: undefined }),
      ];
      expect(computeStats(list).averageRating).toBeNull();
    });
  });

  describe("filterSubmissions", () => {
    const list = [
      makeSubmission({
        listing_version_id: "a",
        status: SubmissionStatus.APPROVED,
      }),
      makeSubmission({
        listing_version_id: "b",
        status: SubmissionStatus.PENDING,
      }),
    ];

    test("returns all when filter is 'all'", () => {
      expect(filterSubmissions(list, "all")).toHaveLength(2);
    });

    test("returns only matching status", () => {
      const result = filterSubmissions(list, SubmissionStatus.PENDING);
      expect(result).toHaveLength(1);
      expect(result[0].listing_version_id).toBe("b");
    });

    test("returns a copy on 'all' so callers can't mutate the input", () => {
      const result = filterSubmissions(list, "all");
      expect(result).not.toBe(list);
      expect(result).toEqual(list);
    });
  });

  describe("formatRuns", () => {
    test("formats sub-thousand values with locale separators", () => {
      expect(formatRuns(0)).toBe("0");
      expect(formatRuns(999)).toBe("999");
    });

    test("formats thousands with K suffix", () => {
      expect(formatRuns(1_500)).toBe("1.5K");
      expect(formatRuns(12_345)).toBe("12.3K");
    });

    test("formats millions with M suffix", () => {
      expect(formatRuns(1_500_000)).toBe("1.5M");
    });

    test("promotes near-million boundary into M (no '1000.0K')", () => {
      expect(formatRuns(999_950)).toBe("1.0M");
      expect(formatRuns(999_949)).toBe("999.9K");
    });
  });

  describe("formatSubmittedAt", () => {
    test("returns em dash for null/undefined", () => {
      expect(formatSubmittedAt(null)).toBe("—");
      expect(formatSubmittedAt(undefined)).toBe("—");
    });

    test("returns em dash for invalid date", () => {
      expect(formatSubmittedAt(new Date("not-a-date"))).toBe("—");
    });

    test("formats valid Date instance", () => {
      const formatted = formatSubmittedAt(new Date("2026-04-15T00:00:00Z"));
      expect(formatted).not.toBe("—");
      expect(formatted.length).toBeGreaterThan(0);
    });
  });

  describe("static option lists", () => {
    test("STATUS_OPTIONS covers all submission statuses except DRAFT visual aside", () => {
      const values = STATUS_OPTIONS.map((o) => o.value);
      expect(values).toContain(SubmissionStatus.PENDING);
      expect(values).toContain(SubmissionStatus.APPROVED);
      expect(values).toContain(SubmissionStatus.REJECTED);
      expect(values).toContain(SubmissionStatus.DRAFT);
    });

    test("STATUS_FILTERS includes 'all' plus every status", () => {
      const values = STATUS_FILTERS.map((f) => f.value);
      expect(values[0]).toBe("all");
      expect(values).toContain(SubmissionStatus.PENDING);
      expect(values).toContain(SubmissionStatus.APPROVED);
      expect(values).toContain(SubmissionStatus.REJECTED);
      expect(values).toContain(SubmissionStatus.DRAFT);
    });

    test("STATUS_VISUAL has an entry for every SubmissionStatus", () => {
      for (const status of Object.values(SubmissionStatus)) {
        expect(STATUS_VISUAL[status]).toBeDefined();
        expect(STATUS_VISUAL[status].label.length).toBeGreaterThan(0);
      }
    });
  });

  describe("getStatusVisual", () => {
    test("returns the matching visual for a known status", () => {
      expect(getStatusVisual(SubmissionStatus.APPROVED)).toBe(
        STATUS_VISUAL[SubmissionStatus.APPROVED],
      );
      expect(getStatusVisual(SubmissionStatus.PENDING)).toBe(
        STATUS_VISUAL[SubmissionStatus.PENDING],
      );
    });

    test("falls back to DRAFT visual for an unknown status", () => {
      const unknown = "ARCHIVED" as unknown as SubmissionStatus;
      expect(getStatusVisual(unknown)).toBe(
        STATUS_VISUAL[SubmissionStatus.DRAFT],
      );
    });
  });
});
