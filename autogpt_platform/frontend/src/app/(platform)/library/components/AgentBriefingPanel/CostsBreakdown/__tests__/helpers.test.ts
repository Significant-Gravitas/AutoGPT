import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { UserDailyCost } from "@/app/api/__generated__/models/userDailyCost";

import {
  buildAgentLookup,
  fillDailyGaps,
  formatRelativeDate,
  formatShortDate,
} from "../helpers";

function daily(date: string, cost = 100, runs = 1): UserDailyCost {
  return {
    date: date as unknown as Date,
    cost_cents: cost,
    run_count: runs,
  };
}

describe("fillDailyGaps", () => {
  test("returns input unchanged when empty", () => {
    expect(fillDailyGaps([])).toEqual([]);
  });

  test("fills missing UTC days with zero buckets between min and max", () => {
    const result = fillDailyGaps([
      daily("2026-05-10", 500),
      daily("2026-05-13", 200),
    ]);

    expect(result.map((d) => d.date)).toEqual([
      "2026-05-10",
      "2026-05-11",
      "2026-05-12",
      "2026-05-13",
    ]);
    expect(result[1]).toEqual({
      date: "2026-05-11" as unknown as Date,
      cost_cents: 0,
      run_count: 0,
    });
    expect(result[3].cost_cents).toBe(200);
  });

  test("normalises Date instances to ISO strings", () => {
    const result = fillDailyGaps([
      {
        date: new Date(Date.UTC(2026, 4, 10)),
        cost_cents: 100,
        run_count: 1,
      },
    ]);

    expect(result[0].date).toBe("2026-05-10");
  });

  test("returns input unchanged when first date is unparseable", () => {
    const bad = [daily("not-a-date")];
    expect(fillDailyGaps(bad)).toBe(bad);
  });
});

describe("buildAgentLookup", () => {
  test("maps each library agent by graph_id", () => {
    const agents = [
      {
        id: "lib-1",
        graph_id: "graph-a",
        name: "Agent A",
        image_url: "https://example.test/a.png",
      },
      {
        id: "lib-2",
        graph_id: "graph-b",
        name: "Agent B",
        image_url: null,
      },
    ] as unknown as LibraryAgent[];

    const lookup = buildAgentLookup(agents);

    expect(lookup.size).toBe(2);
    expect(lookup.get("graph-a")).toEqual({
      libraryAgentId: "lib-1",
      name: "Agent A",
      imageUrl: "https://example.test/a.png",
    });
    expect(lookup.get("graph-b")?.imageUrl).toBeNull();
  });

  test("returns an empty map for no agents", () => {
    expect(buildAgentLookup([]).size).toBe(0);
  });
});

describe("formatRelativeDate", () => {
  const NOW = new Date("2026-05-15T12:00:00.000Z");

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("returns 'just now' for <1 minute ago", () => {
    expect(formatRelativeDate(new Date(NOW.getTime() - 20_000))).toBe(
      "just now",
    );
  });

  test("returns minutes for <1 hour ago", () => {
    expect(formatRelativeDate(new Date(NOW.getTime() - 5 * 60_000))).toBe(
      "5m ago",
    );
  });

  test("returns hours for <1 day ago", () => {
    expect(formatRelativeDate(new Date(NOW.getTime() - 3 * 3_600_000))).toBe(
      "3h ago",
    );
  });

  test("returns days for <30 days ago", () => {
    expect(formatRelativeDate(new Date(NOW.getTime() - 4 * 86_400_000))).toBe(
      "4d ago",
    );
  });

  test("returns absolute month+day when older than 30 days", () => {
    const out = formatRelativeDate(new Date("2026-03-10T12:00:00.000Z"));
    expect(out).toMatch(/Mar/);
    expect(out).toMatch(/10/);
  });

  test("accepts ISO string input", () => {
    expect(formatRelativeDate("2026-05-15T11:50:00.000Z")).toBe("10m ago");
  });
});

describe("formatShortDate", () => {
  test("formats ISO date string as month+day", () => {
    const out = formatShortDate("2026-05-10");
    expect(out).toMatch(/May/);
    expect(out).toMatch(/10/);
  });

  test("formats a Date instance", () => {
    const out = formatShortDate(new Date(Date.UTC(2026, 4, 10)));
    expect(out).toMatch(/May/);
  });

  test("returns the raw input when unparseable", () => {
    expect(formatShortDate("not-a-date")).toBe("not-a-date");
  });
});
