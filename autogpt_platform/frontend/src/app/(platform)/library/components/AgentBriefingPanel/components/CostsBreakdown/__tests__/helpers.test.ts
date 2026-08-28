import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getTenantEntityKey } from "@/services/org-team/identity";

import { buildAgentLookup, formatRelativeDate } from "../helpers";

describe("buildAgentLookup", () => {
  test("maps each library agent by graph and tenant scope", () => {
    const agents = [
      {
        id: "lib-1",
        graph_id: "graph-a",
        name: "Agent A",
        image_url: "https://example.test/a.png",
        organization_id: "org-a",
        team_id: "team-a",
      },
      {
        id: "lib-2",
        graph_id: "graph-b",
        name: "Agent B",
        image_url: null,
        organization_id: "org-a",
        team_id: "team-b",
      },
    ] as unknown as LibraryAgent[];

    const lookup = buildAgentLookup(agents);

    expect(lookup.size).toBe(2);
    expect(
      lookup.get(getTenantEntityKey("graph-a", "org-a", "team-a")),
    ).toEqual({
      libraryAgentId: "lib-1",
      name: "Agent A",
      imageUrl: "https://example.test/a.png",
      organizationId: "org-a",
      teamId: "team-a",
    });
    expect(
      lookup.get(getTenantEntityKey("graph-b", "org-a", "team-b"))?.imageUrl,
    ).toBeNull();
  });

  test("keeps duplicate graph installs in different teams distinct", () => {
    const agents = [
      {
        id: "lib-a",
        graph_id: "shared-graph",
        name: "Team A copy",
        organization_id: "org-1",
        team_id: "team-a",
      },
      {
        id: "lib-b",
        graph_id: "shared-graph",
        name: "Team B copy",
        organization_id: "org-1",
        team_id: "team-b",
      },
    ] as unknown as LibraryAgent[];

    const lookup = buildAgentLookup(agents);

    expect(lookup.size).toBe(2);
    expect(
      lookup.get(getTenantEntityKey("shared-graph", "org-1", "team-a"))
        ?.libraryAgentId,
    ).toBe("lib-a");
    expect(
      lookup.get(getTenantEntityKey("shared-graph", "org-1", "team-b"))
        ?.libraryAgentId,
    ).toBe("lib-b");
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
