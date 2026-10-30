import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { describe, expect, test, vi } from "vitest";
import { getPageView, waitUntilAnsweredAsOpen } from "../helpers";

const SETTINGS = { intervalMS: 0, streak: 2, maxProbes: 10 };
const CLOSED = new ApiError("This account is closed.", 401, {
  detail: "This account is closed.",
});

describe("waitUntilAnsweredAsOpen", () => {
  test("needs the streak in a row: a closed answer starts it over", async () => {
    const probe = vi
      .fn()
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(CLOSED)
      .mockResolvedValue({});

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => false,
      settings: SETTINGS,
    });

    expect(probe).toHaveBeenCalledTimes(4);
  });

  test("another refusal still means the server sees the account as open", async () => {
    const probe = vi
      .fn()
      .mockRejectedValue(new ApiError("Payment required", 402, {}));

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => false,
      settings: SETTINGS,
    });

    expect(probe).toHaveBeenCalledTimes(2);
  });

  test("a network failure is not an answer, and it gives up after maxProbes", async () => {
    const probe = vi.fn().mockRejectedValue(new TypeError("fetch failed"));

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => false,
      settings: SETTINGS,
    });

    expect(probe).toHaveBeenCalledTimes(10);
  });

  test("stops when cancelled", async () => {
    const probe = vi.fn().mockRejectedValue(CLOSED);

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => probe.mock.calls.length >= 1,
      settings: SETTINGS,
    });

    expect(probe).toHaveBeenCalledTimes(1);
  });
});

describe("getPageView", () => {
  const base = {
    isKept: false,
    isLoading: false,
    isError: false,
    now: new Date("2026-11-13T12:00:00Z"),
  };

  test("a closed account before its erasure date can be kept", () => {
    expect(
      getPageView({
        ...base,
        status: "scheduled",
        eraseAfter: "2026-11-13T12:00:01Z",
      }),
    ).toBe("closed");
  });

  test("from the erasure moment on it is too late", () => {
    expect(
      getPageView({
        ...base,
        status: "scheduled",
        eraseAfter: "2026-11-13T12:00:00Z",
      }),
    ).toBe("too_late");
  });

  test("a kept or open account is open", () => {
    expect(
      getPageView({
        ...base,
        isKept: true,
        status: "scheduled",
        eraseAfter: null,
      }),
    ).toBe("open");
    expect(getPageView({ ...base, status: "open", eraseAfter: null })).toBe(
      "open",
    );
  });
});
