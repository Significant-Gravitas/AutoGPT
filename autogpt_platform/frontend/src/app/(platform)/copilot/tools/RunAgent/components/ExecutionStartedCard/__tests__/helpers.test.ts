import { describe, expect, it } from "vitest";
import type { getV1GetExecutionDetailsResponse } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  LIVE_STATUS_POLL_MS,
  nextPollInterval,
  retryUnlessPermanent,
  shouldWatchRun,
} from "../helpers";

function ok(status: AgentExecutionStatus) {
  return { status: 200, data: { status } } as getV1GetExecutionDetailsResponse;
}

function state(
  data: getV1GetExecutionDetailsResponse | undefined,
  polls = 1,
  error: unknown = null,
) {
  return { data, error, dataUpdateCount: polls, errorUpdateCount: 0 };
}

describe("nextPollInterval", () => {
  it("polls every 5 s for the first minute", () => {
    expect(nextPollInterval(state(ok("RUNNING"), 1))).toBe(LIVE_STATUS_POLL_MS);
    expect(nextPollInterval(state(ok("RUNNING"), 11))).toBe(
      LIVE_STATUS_POLL_MS,
    );
  });

  it("backs off to at most a minute for a run that stays open", () => {
    expect(nextPollInterval(state(ok("REVIEW"), 12))).toBe(
      LIVE_STATUS_POLL_MS * 2,
    );
    expect(nextPollInterval(state(ok("REVIEW"), 500))).toBe(60_000);
  });

  it.each(["COMPLETED", "FAILED", "TERMINATED"] as const)(
    "stops at %s",
    (status) => {
      expect(nextPollInterval(state(ok(status)))).toBe(false);
    },
  );

  it("stays stopped for an ended run when a later refetch fails", () => {
    expect(
      nextPollInterval(state(ok("FAILED"), 3, new ApiError("down", 502, null))),
    ).toBe(false);
  });

  it("stops on a 404 but keeps going on a 401 or a 5xx", () => {
    expect(
      nextPollInterval(
        state(ok("RUNNING"), 2, new ApiError("gone", 404, null)),
      ),
    ).toBe(false);
    expect(
      nextPollInterval(state(undefined, 1, new ApiError("auth", 401, null))),
    ).toBe(LIVE_STATUS_POLL_MS);
    expect(
      nextPollInterval(state(undefined, 1, new ApiError("down", 502, null))),
    ).toBe(LIVE_STATUS_POLL_MS);
  });
});

describe("retryUnlessPermanent", () => {
  it("doesn't retry a 404 or 403, but retries a 5xx up to three times", () => {
    expect(retryUnlessPermanent(0, new ApiError("gone", 404, null))).toBe(
      false,
    );
    expect(retryUnlessPermanent(0, new ApiError("no", 403, null))).toBe(false);
    expect(retryUnlessPermanent(0, new ApiError("down", 502, null))).toBe(true);
    expect(retryUnlessPermanent(3, new ApiError("down", 502, null))).toBe(
      false,
    );
  });
});

describe("shouldWatchRun", () => {
  it.each([
    ["QUEUED", true],
    ["RUNNING", true],
    ["REVIEW", true],
    ["unknown", true],
    [undefined, true],
    ["SCHEDULED", false],
    ["COMPLETED", false],
    ["failed", false],
  ])("%s -> %s", (status, expected) => {
    expect(shouldWatchRun(status)).toBe(expected);
  });
});
