import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

import {
  findPublishedSubmission,
  formatRelativeDate,
  formatSubmissionDate,
  getSubmissionMeta,
} from "../helpers";

const NOW = new Date("2026-07-04T12:00:00Z");
const DAY = 86_400_000;

function daysAgo(days: number): string {
  return new Date(NOW.getTime() - days * DAY).toISOString();
}

describe("PublishAgentModal helpers", () => {
  describe("formatSubmissionDate", () => {
    it("returns null for empty or invalid values", () => {
      expect(formatSubmissionDate(null)).toBeNull();
      expect(formatSubmissionDate(undefined)).toBeNull();
      expect(formatSubmissionDate("not-a-date")).toBeNull();
    });

    it("formats a valid date", () => {
      expect(formatSubmissionDate("2026-07-01T00:00:00Z")).toBeTruthy();
    });
  });

  describe("formatRelativeDate", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(NOW);
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it("returns null for empty or invalid values", () => {
      expect(formatRelativeDate(null)).toBeNull();
      expect(formatRelativeDate("nope")).toBeNull();
    });

    it("covers each relative bucket", () => {
      expect(formatRelativeDate(daysAgo(0))).toBe("today");
      expect(formatRelativeDate(daysAgo(-2))).toBe("today");
      expect(formatRelativeDate(daysAgo(1))).toBe("yesterday");
      expect(formatRelativeDate(daysAgo(3))).toBe("3 days ago");
      expect(formatRelativeDate(daysAgo(8))).toBe("1 week ago");
      expect(formatRelativeDate(daysAgo(15))).toBe("2 weeks ago");
      expect(formatRelativeDate(daysAgo(40))).toBe("1 month ago");
      expect(formatRelativeDate(daysAgo(70))).toBe("2 months ago");
      expect(formatRelativeDate(daysAgo(400))).toBe("1 year ago");
      expect(formatRelativeDate(daysAgo(800))).toBe("2 years ago");
    });
  });

  describe("getSubmissionMeta", () => {
    const base = {
      version: undefined,
      category: undefined,
      submittedAt: undefined,
      reviewedAt: undefined,
      runCount: undefined,
    };

    it("includes version and category when present", () => {
      const items = getSubmissionMeta({
        ...base,
        status: SubmissionStatus.PENDING,
        version: 4,
        category: "Marketing",
      });
      expect(items).toContainEqual({ label: "Version", value: "v4" });
      expect(items).toContainEqual({ label: "Category", value: "Marketing" });
    });

    it("shows live-since date and runs for approved", () => {
      const items = getSubmissionMeta({
        ...base,
        status: SubmissionStatus.APPROVED,
        reviewedAt: "2026-07-02T00:00:00Z",
        runCount: 4200,
      });
      expect(items.some((i) => i.label === "Live since")).toBe(true);
      expect(items).toContainEqual({ label: "Runs", value: "4,200" });
    });

    it("shows reviewed date for rejected", () => {
      const items = getSubmissionMeta({
        ...base,
        status: SubmissionStatus.REJECTED,
        reviewedAt: "2026-07-02T00:00:00Z",
      });
      expect(items.some((i) => i.label === "Reviewed")).toBe(true);
    });

    it("shows 'Not submitted yet' for draft", () => {
      const items = getSubmissionMeta({
        ...base,
        status: SubmissionStatus.DRAFT,
      });
      expect(items).toContainEqual({
        label: "Status",
        value: "Not submitted yet",
      });
    });

    it("shows submitted date for pending", () => {
      const items = getSubmissionMeta({
        ...base,
        status: SubmissionStatus.PENDING,
        submittedAt: "2026-07-01T00:00:00Z",
      });
      expect(items.some((i) => i.label === "Submitted")).toBe(true);
    });
  });

  describe("findPublishedSubmission", () => {
    function makeSubmission(
      overrides: Partial<StoreSubmission>,
    ): StoreSubmission {
      return {
        listing_id: "listing-1",
        user_id: "user-1",
        slug: "listed-slug",
        listing_version_id: "listing-version-1",
        listing_version: 1,
        graph_id: "g1",
        graph_version: 1,
        name: "Listed name",
        sub_heading: "Listed subheading",
        description: "Listed description",
        instructions: null,
        categories: [],
        image_urls: [],
        video_url: null,
        agent_output_demo_url: null,
        submitted_at: null,
        changes_summary: null,
        status: SubmissionStatus.APPROVED,
        ...overrides,
      };
    }

    it("returns the graph's newest approved listing version, the live one", () => {
      // Listing version 3 relisted an older graph version than version 2 did.
      const live = makeSubmission({ listing_version: 3, graph_version: 1 });
      const submissions = [
        makeSubmission({ listing_version: 1, graph_version: 1 }),
        makeSubmission({ listing_version: 2, graph_version: 2 }),
        live,
        makeSubmission({
          listing_version: 4,
          graph_version: 3,
          status: SubmissionStatus.PENDING,
        }),
        makeSubmission({
          graph_id: "g2",
          listing_version: 5,
          graph_version: 5,
        }),
      ];

      expect(findPublishedSubmission(submissions, "g1")).toBe(live);
    });

    it("returns undefined when the graph has no approved submission", () => {
      const submissions = [
        makeSubmission({ status: SubmissionStatus.REJECTED }),
        makeSubmission({ graph_id: "g2" }),
      ];

      expect(findPublishedSubmission(submissions, "g1")).toBeUndefined();
      expect(findPublishedSubmission(undefined, "g1")).toBeUndefined();
    });
  });
});
