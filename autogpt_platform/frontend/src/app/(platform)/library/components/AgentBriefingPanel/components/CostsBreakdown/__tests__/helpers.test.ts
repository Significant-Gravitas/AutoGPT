import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

import { buildAgentLookup, formatRelativeDate } from "../helpers";

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
