import { describe, expect, it } from "vitest";
import { isAgentScheduled } from "./executionHelpers";

describe("isAgentScheduled", () => {
  it("returns true when is_scheduled is set", () => {
    expect(
      isAgentScheduled({ is_scheduled: true, recommended_schedule_cron: null }),
    ).toBe(true);
  });

  it("returns true when recommended_schedule_cron is a non-empty string", () => {
    expect(
      isAgentScheduled({
        is_scheduled: false,
        recommended_schedule_cron: "0 9 * * *",
      }),
    ).toBe(true);
  });

  it("returns true when both flags are set", () => {
    expect(
      isAgentScheduled({
        is_scheduled: true,
        recommended_schedule_cron: "0 9 * * *",
      }),
    ).toBe(true);
  });

  it("returns false when neither flag is set", () => {
    expect(
      isAgentScheduled({
        is_scheduled: false,
        recommended_schedule_cron: null,
      }),
    ).toBe(false);
  });

  it("returns false when both fields are undefined", () => {
    expect(isAgentScheduled({})).toBe(false);
  });

  it("returns false when recommended_schedule_cron is an empty string", () => {
    expect(
      isAgentScheduled({
        is_scheduled: false,
        recommended_schedule_cron: "",
      }),
    ).toBe(false);
  });
});
