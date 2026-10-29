import { describe, expect, it } from "vitest";
import { isAgentScheduled } from "./executionHelpers";

describe("isAgentScheduled", () => {
  it("returns true when is_scheduled is set", () => {
    expect(isAgentScheduled({ is_scheduled: true })).toBe(true);
  });

  it("returns false when is_scheduled is false", () => {
    expect(isAgentScheduled({ is_scheduled: false })).toBe(false);
  });

  it("returns false when is_scheduled is undefined", () => {
    expect(isAgentScheduled({})).toBe(false);
  });

  it("ignores recommended_schedule_cron (creator suggestion, not a user schedule)", () => {
    const agentWithRecommendation = {
      is_scheduled: false,
      recommended_schedule_cron: "0 9 * * *",
    };
    expect(isAgentScheduled(agentWithRecommendation)).toBe(false);
  });
});
