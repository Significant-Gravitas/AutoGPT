import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  logClientRequestFailure,
  resetClientRequestFailureLog,
} from "../request-failure-log";

const base = {
  method: "GET",
  url: "/api/credits/subscription",
  errorMessage: "Backend sent no response within 30000ms",
  responseData: null,
};

describe("logClientRequestFailure", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    resetClientRequestFailureLog();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("logs the first failure and suppresses repeats of the same method, url and status", () => {
    for (let i = 0; i < 12; i++) {
      logClientRequestFailure({ ...base, status: 504 });
    }

    expect(console.error).toHaveBeenCalledTimes(1);
    expect(console.error).toHaveBeenCalledWith(
      "Request failed on client",
      expect.objectContaining({ status: 504, url: base.url }),
    );
  });

  it("reports how many were suppressed once the window closes", () => {
    logClientRequestFailure({ ...base, status: 504 });
    logClientRequestFailure({ ...base, status: 504 });
    logClientRequestFailure({ ...base, status: 504 });

    expect(console.error).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(10_000);

    expect(console.error).toHaveBeenCalledTimes(2);
    expect(console.error).toHaveBeenLastCalledWith(
      "Request failed on client ×3 in the last 10s",
      expect.objectContaining({ status: 504, url: base.url }),
    );
    expect(console.warn).not.toHaveBeenCalled();
  });

  it("reports suppressed expected failures at warn", () => {
    logClientRequestFailure({ ...base, status: 404 });
    logClientRequestFailure({ ...base, status: 404 });
    vi.advanceTimersByTime(10_000);

    expect(console.warn).toHaveBeenLastCalledWith(
      "Request failed on client ×2 in the last 10s",
      expect.objectContaining({ status: 404, url: base.url }),
    );
    expect(console.error).not.toHaveBeenCalled();
  });

  it("says nothing extra when the failure happened once", () => {
    logClientRequestFailure({ ...base, status: 504 });
    vi.advanceTimersByTime(10_000);

    expect(console.warn).not.toHaveBeenCalled();
  });

  it("logs again once the window has passed", () => {
    logClientRequestFailure({ ...base, status: 504 });
    vi.advanceTimersByTime(10_001);
    logClientRequestFailure({ ...base, status: 504 });

    expect(console.error).toHaveBeenCalledTimes(2);
  });

  it("keeps a different status, url or method visible", () => {
    logClientRequestFailure({ ...base, status: 504 });
    logClientRequestFailure({ ...base, status: 500 });
    logClientRequestFailure({ ...base, status: 504, url: "/api/other" });
    logClientRequestFailure({ ...base, status: 504, method: "POST" });

    expect(console.error).toHaveBeenCalledTimes(4);
  });

  it("warns rather than errors on the statuses the app asks for and handles", () => {
    logClientRequestFailure({ ...base, status: 401 });
    logClientRequestFailure({ ...base, status: 403 });
    logClientRequestFailure({ ...base, status: 404 });

    expect(console.error).not.toHaveBeenCalled();
    expect(console.warn).toHaveBeenCalledTimes(3);
  });

  it("keeps a malformed request an error", () => {
    logClientRequestFailure({ ...base, status: 422 });

    expect(console.error).toHaveBeenCalledTimes(1);
  });
});
