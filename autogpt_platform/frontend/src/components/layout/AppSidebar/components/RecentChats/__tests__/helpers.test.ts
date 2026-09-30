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

  it("labels an older same-year date with month and day", () => {
    // 2026-06-20 -> "June 20"
    const label = getDateGroupLabel("2026-06-20T08:00:00", "en-US");
    expect(label).toBe("June 20");
  });

  it("includes the year for dates in a previous year", () => {
    const label = getDateGroupLabel("2024-12-01T08:00:00", "en-US");
    expect(label).toBe("December 1, 2024");
  });

  it("formats the date end-to-end in the user's locale", () => {
    // The label must come from a single locale-aware formatter, not from
    // hand-assembled English-style parts ("20th 六月"-style mixes).
    expect(getDateGroupLabel("2026-06-20T08:00:00", "en-GB")).toBe("20 June");
    expect(getDateGroupLabel("2026-06-20T08:00:00", "de-DE")).toBe("20. Juni");
    expect(getDateGroupLabel("2026-06-20T08:00:00", "zh-CN")).toBe("6月20日");
    expect(getDateGroupLabel("2024-12-01T08:00:00", "zh-CN")).toBe(
      "2024年12月1日",
    );
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
    const groups = groupSessionsByDate(sessions, "en-US");
    expect(groups.map((g) => g.label)).toEqual([
      "Today",
      "Yesterday",
      getDateGroupLabel(isoDaysAgo(5), "en-US"),
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

  it("labels groups in the given locale", () => {
    const groups = groupSessionsByDate(
      [{ id: "a", updated_at: "2026-06-20T08:00:00" }],
      "de-DE",
    );
    expect(groups[0].label).toBe("20. Juni");
  });
});
