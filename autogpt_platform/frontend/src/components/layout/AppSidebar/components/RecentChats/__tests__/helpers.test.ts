import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getDateGroupLabel, groupSessionsByDate } from "../helpers";

// Pin "now" so the day-relative labels are deterministic.
const NOW = new Date("2026-06-30T12:00:00Z");

function isoDaysAgo(days: number): string {
  const d = new Date(NOW);
  d.setDate(d.getDate() - days);
  return d.toISOString();
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("getDateGroupLabel", () => {
  it("labels today's date as 'Today'", () => {
    expect(getDateGroupLabel(isoDaysAgo(0))).toBe("Today");
  });

  it("labels a future-ish (negative diff) date as 'Today'", () => {
    const future = new Date(NOW);
    future.setHours(future.getHours() + 5);
    expect(getDateGroupLabel(future.toISOString())).toBe("Today");
  });

  it("labels yesterday as 'Yesterday'", () => {
    expect(getDateGroupLabel(isoDaysAgo(1))).toBe("Yesterday");
  });

  it("labels an older same-year date with an ordinal day and month", () => {
    // 2026-06-20 -> "20th June"
    const label = getDateGroupLabel("2026-06-20T08:00:00");
    expect(label).toBe("20th June");
  });

  it("includes the year for dates in a previous year", () => {
    const label = getDateGroupLabel("2024-12-01T08:00:00");
    expect(label).toBe("1st December 2024");
  });

  it("uses 'st', 'nd', 'rd', 'th' ordinal suffixes correctly", () => {
    expect(getDateGroupLabel("2026-06-01T08:00:00")).toBe("1st June");
    expect(getDateGroupLabel("2026-06-02T08:00:00")).toBe("2nd June");
    expect(getDateGroupLabel("2026-06-03T08:00:00")).toBe("3rd June");
    expect(getDateGroupLabel("2026-06-04T08:00:00")).toBe("4th June");
  });

  it("uses 'th' for the 11th–13th teen exceptions", () => {
    expect(getDateGroupLabel("2026-06-11T08:00:00")).toBe("11th June");
    expect(getDateGroupLabel("2026-06-12T08:00:00")).toBe("12th June");
    expect(getDateGroupLabel("2026-06-13T08:00:00")).toBe("13th June");
  });
});

describe("groupSessionsByDate", () => {
  it("returns an empty array for no sessions", () => {
    expect(groupSessionsByDate([])).toEqual([]);
  });

  it("buckets sessions from the same calendar day into one group", () => {
    // Local-time (no trailing Z) so the day boundary matches the runner's
    // timezone — startOfDay() uses local getFullYear/Month/Date.
    const sessions = [
      { id: "a", updated_at: "2026-06-30T01:00:00" },
      { id: "b", updated_at: "2026-06-30T23:00:00" },
    ];
    const groups = groupSessionsByDate(sessions);
    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe("Today");
    expect(groups[0].sessions.map((s) => s.id)).toEqual(["a", "b"]);
  });

  it("orders groups most-recent-day first regardless of input order", () => {
    const sessions = [
      { id: "old", updated_at: isoDaysAgo(5) },
      { id: "today", updated_at: isoDaysAgo(0) },
      { id: "yesterday", updated_at: isoDaysAgo(1) },
    ];
    const groups = groupSessionsByDate(sessions);
    expect(groups.map((g) => g.label)).toEqual([
      "Today",
      "Yesterday",
      getDateGroupLabel(isoDaysAgo(5)),
    ]);
  });

  it("preserves input order within a group and never duplicates day buckets", () => {
    const sessions = [
      { id: "1", updated_at: "2026-06-28T03:00:00" },
      { id: "2", updated_at: "2026-06-28T20:00:00" },
      { id: "3", updated_at: "2026-06-28T11:00:00" },
    ];
    const groups = groupSessionsByDate(sessions);
    expect(groups).toHaveLength(1);
    expect(groups[0].sessions.map((s) => s.id)).toEqual(["1", "2", "3"]);
  });
});
