import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_COLUMNS,
  DONE,
  IN_PROGRESS,
  buildScheduleCards,
  formatRunMoment,
  formatTimeUntil,
  getCardColumnId,
  getNextFollowupBySession,
  getPreviewLine,
  makeColumnId,
  parseBoardState,
} from "../helpers";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";

const now = Date.parse("2026-10-16T12:00:00Z");
const old: SessionSummaryResponse = {
  id: "chat",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  is_processing: false,
};
const columns = new Set([IN_PROGRESS, DONE, "custom"]);

it("ignores a manual placement while working, after activity, or after column removal", () => {
  const placement = { columnId: "custom", updatedAt: old.updated_at };
  expect(getCardColumnId(old, placement, columns, now)).toBe("custom");
  for (const chat_status of ["running", "queued"]) {
    expect(
      getCardColumnId({ ...old, chat_status }, placement, columns, now),
    ).toBe(IN_PROGRESS);
  }
  expect(
    getCardColumnId({ ...old, is_processing: true }, placement, columns, now),
  ).toBe(IN_PROGRESS);
  expect(
    getCardColumnId(
      { ...old, updated_at: new Date(now).toISOString() },
      placement,
      columns,
      now,
    ),
  ).toBe(IN_PROGRESS);
  expect(
    getCardColumnId(old, placement, new Set([IN_PROGRESS, DONE]), now),
  ).toBe(DONE);
});

it("restores custom columns and handles missing or invalid stored JSON", () => {
  for (const raw of [null, "invalid", "null"]) {
    expect(parseBoardState(raw)).toEqual({
      columns: DEFAULT_COLUMNS,
      placements: {},
    });
  }
  expect(
    parseBoardState(
      JSON.stringify({
        columns: [...DEFAULT_COLUMNS, { id: "custom", name: "Later" }],
      }),
    ).columns,
  ).toHaveLength(3);
  expect(makeColumnId("私の仕事")).toMatch(/^column-/);
  expect(getPreviewLine("## **Summary**")).toBe("Summary");
});

const followup: CopilotTurnJobInfo = {
  id: "followup",
  name: "Follow up",
  kind: "copilot_turn",
  user_id: "user",
  message: "**Prepare a report**",
  session_id: "chat",
  next_run_time: "2026-10-17T12:00:00Z",
};

it("sorts schedules and picks the earliest follow-up for each chat", () => {
  const earlier = {
    ...followup,
    id: "earlier",
    next_run_time: "2026-10-16T13:00:00Z",
  };
  const unscheduled = { ...followup, id: "paused", next_run_time: "" };
  const newChat = { ...followup, id: "new", session_id: null };
  const cards = buildScheduleCards(
    [followup, earlier, unscheduled, newChat],
    [],
  );
  expect(cards[0].id).toBe("earlier");
  expect(cards[0].title).toBe("Prepare a report");
  expect(cards[0].href).toBe("/home?sessionId=chat");
  expect(cards.find((card) => card.id === "new")?.href).toBe(
    "/library/followups",
  );
  expect(cards.at(-1)?.nextRunTime).toBeNull();
  expect(
    getNextFollowupBySession([followup, earlier, unscheduled, newChat]).get(
      "chat",
    ),
  ).toBe(earlier.next_run_time);
});

describe("schedule times", () => {
  it.each([
    [0, "now"],
    [30, "in 30m"],
    [120, "in 2h"],
    [2880, "in 2d"],
  ])("formats %s minutes", (minutes, label) => {
    expect(
      formatTimeUntil(
        new Date(now + Number(minutes) * 60_000).toISOString(),
        now,
      ),
    ).toBe(label);
  });
  it("handles missing dates and formats a scheduled moment", () => {
    expect(formatTimeUntil(null)).toBeNull();
    expect(formatTimeUntil("invalid")).toBeNull();
    expect(formatRunMoment(followup.next_run_time, "en-US")).toContain("Sat");
  });
});
