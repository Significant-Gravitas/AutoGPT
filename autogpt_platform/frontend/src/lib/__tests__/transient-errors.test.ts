import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { afterEach, describe, expect, it, vi } from "vitest";
import { isTransientRequestError, retryTransient } from "../transient-errors";

function apiError(status: number) {
  return new ApiError(`Request failed with ${status}`, status, {});
}

afterEach(() => {
  vi.useRealTimers();
});

describe("isTransientRequestError", () => {
  it.each([500, 502, 503, 504, 408, 425, 429])(
    "counts an API error with status %i as worth retrying",
    (status) => {
      expect(isTransientRequestError(apiError(status))).toBe(true);
    },
  );

  it.each([400, 401, 403, 404, 409, 422, 499])(
    "counts an API error with status %i as an answer",
    (status) => {
      expect(isTransientRequestError(apiError(status))).toBe(false);
    },
  );

  it.each([
    "Failed to fetch",
    "Load failed",
    "NetworkError when attempting to fetch resource.",
    "fetch failed",
  ])("counts a dropped connection (%s) as worth retrying", (message) => {
    expect(isTransientRequestError(new TypeError(message))).toBe(true);
  });

  it("counts a cancelled request as worth retrying later", () => {
    expect(
      isTransientRequestError(new DOMException("Aborted", "AbortError")),
    ).toBe(true);
  });

  it("counts any other error as a real failure", () => {
    expect(isTransientRequestError(new TypeError("x is not a function"))).toBe(
      false,
    );
    expect(isTransientRequestError(new Error("boom"))).toBe(false);
  });
});

describe("retryTransient", () => {
  it("retries a server error until the request goes through", async () => {
    vi.useFakeTimers();
    const request = vi
      .fn()
      .mockRejectedValueOnce(apiError(503))
      .mockResolvedValue("answer");

    const result = retryTransient(request);
    await vi.advanceTimersByTimeAsync(1_000);

    await expect(result).resolves.toBe("answer");
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("gives up on a server error after the last attempt", async () => {
    vi.useFakeTimers();
    const request = vi.fn().mockRejectedValue(apiError(502));

    const result = retryTransient(request, { attempts: 3 });
    const outcome = expect(result).rejects.toMatchObject({ status: 502 });
    await vi.advanceTimersByTimeAsync(1_000 + 2_000);

    await outcome;
    expect(request).toHaveBeenCalledTimes(3);
  });

  it("doesn't retry an error the server meant", async () => {
    const request = vi.fn().mockRejectedValue(apiError(400));

    await expect(retryTransient(request)).rejects.toMatchObject({
      status: 400,
    });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("doesn't retry a request its caller cancelled", async () => {
    const request = vi
      .fn()
      .mockRejectedValue(new DOMException("Aborted", "AbortError"));

    await expect(retryTransient(request)).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(request).toHaveBeenCalledTimes(1);
  });
});
