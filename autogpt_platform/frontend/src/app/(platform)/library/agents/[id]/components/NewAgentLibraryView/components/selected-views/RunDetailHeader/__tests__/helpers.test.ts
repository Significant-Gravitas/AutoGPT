import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { describe, expect, it } from "vitest";

import { formatRunDuration, isLiveRun } from "../helpers";

const NOW = Date.parse("2026-07-21T12:00:16Z");

// Timestamps arrive as ISO strings over the API, though the generated type says Date.
function makeRun(overrides: Record<string, unknown> = {}): GraphExecution {
  return {
    id: "run-1",
    status: AgentExecutionStatus.RUNNING,
    started_at: "2026-07-21T12:00:00Z",
    ended_at: null,
    stats: null,
    ...overrides,
  } as unknown as GraphExecution;
}

describe("formatRunDuration", () => {
  it("counts a running run up to now, past a stale ended_at and stats.duration", () => {
    const run = makeRun({
      ended_at: "2026-07-21T12:00:03Z",
      stats: { duration: 3 },
    });

    expect(formatRunDuration(run, NOW)).toBe("16s");
  });

  it("counts a queued run that has started up to now", () => {
    const run = makeRun({ status: AgentExecutionStatus.QUEUED });

    expect(formatRunDuration(run, NOW)).toBe("16s");
  });

  it("keeps counting each second past the first minute", () => {
    const run = makeRun();

    expect(formatRunDuration(run, Date.parse("2026-07-21T12:01:30Z"))).toBe(
      "1m 30s",
    );
    expect(formatRunDuration(run, Date.parse("2026-07-21T12:01:31Z"))).toBe(
      "1m 31s",
    );
  });

  it("shows a few seconds, not nothing, while the floored clock lags a run that just started", () => {
    const run = makeRun({ started_at: "2026-07-21T12:00:16.400Z" });

    expect(formatRunDuration(run, NOW)).toBe("a few seconds");
  });

  it("uses stats.duration once the run has finished", () => {
    const run = makeRun({
      status: AgentExecutionStatus.COMPLETED,
      ended_at: "2026-07-21T12:05:00Z",
      stats: { duration: 42 },
    });

    expect(formatRunDuration(run, NOW)).toBe("42 seconds");
  });

  it("falls back to ended_at for a finished run without stats.duration", () => {
    const run = makeRun({
      status: AgentExecutionStatus.FAILED,
      ended_at: "2026-07-21T12:00:30Z",
    });

    expect(formatRunDuration(run, NOW)).toBe("30 seconds");
  });

  it.each([
    [
      "a finished run with neither stats.duration nor ended_at",
      { status: AgentExecutionStatus.TERMINATED },
    ],
    ["a run that has not started", { started_at: null }],
    ["a run with an unreadable start", { started_at: "not a date" }],
    [
      "a finished run that ended before it started",
      {
        status: AgentExecutionStatus.COMPLETED,
        ended_at: "2026-07-21T11:59:00Z",
      },
    ],
  ])("shows no duration for %s", (_, overrides) => {
    const run = makeRun(overrides);

    expect(formatRunDuration(run, NOW)).toBeNull();
  });
});

describe("isLiveRun", () => {
  it.each([
    [AgentExecutionStatus.RUNNING, true],
    [AgentExecutionStatus.QUEUED, true],
    [AgentExecutionStatus.REVIEW, false],
    [AgentExecutionStatus.COMPLETED, false],
  ])("treats %s as live: %s", (status, live) => {
    expect(isLiveRun(makeRun({ status }))).toBe(live);
  });

  it("treats no run as not live", () => {
    expect(isLiveRun(undefined)).toBe(false);
  });
});
