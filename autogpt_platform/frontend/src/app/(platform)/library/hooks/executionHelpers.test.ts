import { describe, expect, it } from "vitest";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import {
  formatRelativeDuration,
  isAgentScheduled,
  runningMessage,
} from "./executionHelpers";

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

describe("formatRelativeDuration", () => {
  it("keeps sub-threshold durations as a few seconds", () => {
    expect(formatRelativeDuration(0)).toBe("a few seconds");
    expect(formatRelativeDuration(4_999)).toBe("a few seconds");
  });

  it("shows second-level precision after the short threshold (16s ≠ a few seconds)", () => {
    expect(formatRelativeDuration(16_000)).toBe("16s");
    expect(formatRelativeDuration(16_000)).not.toBe("a few seconds");
    expect(formatRelativeDuration(59_000)).toBe("59s");
  });

  it("formats minutes and hours like getExecutionDuration", () => {
    expect(formatRelativeDuration(90_000)).toBe("1m 30s");
    expect(formatRelativeDuration(3_600_000)).toBe("1h");
    expect(formatRelativeDuration(3_660_000)).toBe("1h 1m");
  });

  it("formats multi-day durations", () => {
    expect(formatRelativeDuration(90_000_000)).toBe("1d 1h");
  });
});

describe("runningMessage", () => {
  it("uses live elapsed from started_at for RUNNING", () => {
    const started = new Date("2026-09-15T02:00:00.000Z");
    const now = started.getTime() + 16_000;
    expect(runningMessage(AgentExecutionStatus.RUNNING, started, now)).toBe(
      "Running for 16s",
    );
  });
});
