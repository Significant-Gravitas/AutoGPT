import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { describe, expect, it } from "vitest";
import { getAgentStatusLabel, getEmptyMessage } from "../helpers";

function makeAgent(overrides: Partial<LibraryAgent> = {}): LibraryAgent {
  return {
    id: "lib-1",
    graph_id: "g-1",
    name: "Agent One",
    next_scheduled_run: null,
    ...overrides,
  } as unknown as LibraryAgent;
}

describe("getAgentStatusLabel", () => {
  it("returns the static label for non-scheduled tabs", () => {
    expect(getAgentStatusLabel("listening", makeAgent())).toBe(
      "Waiting for trigger event",
    );
    expect(getAgentStatusLabel("idle", makeAgent())).toBe("No recent activity");
  });

  it("returns an empty string for an unknown tab", () => {
    expect(getAgentStatusLabel("nope", makeAgent())).toBe("");
  });

  it("falls back to the static scheduled label when no next run is set", () => {
    expect(getAgentStatusLabel("scheduled", makeAgent())).toBe(
      "Has a scheduled run",
    );
  });

  it("says 'soon' once the scheduled time has passed", () => {
    const past = new Date(Date.now() - 60_000).toISOString();
    expect(
      getAgentStatusLabel("scheduled", makeAgent({ next_scheduled_run: past })),
    ).toBe("Scheduled to run soon");
  });

  it("formats minutes when the next run is under an hour away", () => {
    const in30m = new Date(Date.now() + 30 * 60_000).toISOString();
    expect(
      getAgentStatusLabel(
        "scheduled",
        makeAgent({ next_scheduled_run: in30m }),
      ),
    ).toMatch(/^Scheduled to run in \d+m$/);
  });

  it("formats hours when the next run is under a day away", () => {
    const in5h = new Date(Date.now() + 5 * 60 * 60_000).toISOString();
    expect(
      getAgentStatusLabel("scheduled", makeAgent({ next_scheduled_run: in5h })),
    ).toMatch(/^Scheduled to run in \d+h$/);
  });

  it("formats days when the next run is a day or more away", () => {
    const in3d = new Date(Date.now() + 3 * 24 * 60 * 60_000).toISOString();
    expect(
      getAgentStatusLabel("scheduled", makeAgent({ next_scheduled_run: in3d })),
    ).toMatch(/^Scheduled to run in \d+d$/);
  });
});

describe("getEmptyMessage", () => {
  it("uses task terminology for completed and scheduled tabs", () => {
    expect(getEmptyMessage("completed")).toBe("No recently completed tasks");
    expect(getEmptyMessage("scheduled")).toBe("No agents with scheduled tasks");
  });

  it("returns the matching message for each known tab", () => {
    expect(getEmptyMessage("running")).toBe("No agents running right now");
    expect(getEmptyMessage("attention")).toBe("No agents that need attention");
    expect(getEmptyMessage("listening")).toBe("No agents listening for events");
    expect(getEmptyMessage("idle")).toBe("No idle agents");
  });

  it("falls back to a generic message for an unmapped tab", () => {
    expect(getEmptyMessage("healthy")).toBe("No agents in this category");
  });
});
