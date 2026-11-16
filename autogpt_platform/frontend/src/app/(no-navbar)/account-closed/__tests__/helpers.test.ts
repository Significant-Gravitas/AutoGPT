import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { describe, expect, test, vi } from "vitest";
import { getPageView, waitUntilAnsweredAsOpen } from "../helpers";

const SETTINGS = { intervalMS: 0, streak: 2, minWaitMS: 0, maxProbes: 10 };
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
      keptAt: 0,
    });

    expect(probe).toHaveBeenCalledTimes(4);
  });

  test("waits out every server's cache since the keep, however the probes answer", async () => {
    const probe = vi.fn().mockResolvedValue({});
    let clock = 1_000;

    await waitUntilAnsweredAsOpen({
      probe: async () => {
        clock += 10_000;
        return probe();
      },
      isCancelled: () => false,
      settings: { ...SETTINGS, minWaitMS: 35_000 },
      keptAt: 1_000,
      now: () => clock,
    });

    // Two open answers come at 20 s; it keeps probing until 40 s.
    expect(probe).toHaveBeenCalledTimes(4);
  });

  test.each([
    ["a lost session", 401, { detail: "Not authenticated" }],
    ["a server error", 500, { detail: "boom" }],
  ])("%s is not counted as open", async (_, status, body) => {
    const probe = vi
      .fn()
      .mockRejectedValue(new ApiError("refused", status, body));

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => false,
      settings: SETTINGS,
      keptAt: 0,
    });

    expect(probe).toHaveBeenCalledTimes(SETTINGS.maxProbes);
  });

  test("another refusal still means the server sees the account as open", async () => {
    const probe = vi
      .fn()
      .mockRejectedValue(new ApiError("Payment required", 402, {}));

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => false,
      settings: SETTINGS,
      keptAt: 0,
    });

    expect(probe).toHaveBeenCalledTimes(2);
  });

  test("a network failure is not an answer, and it gives up after maxProbes", async () => {
    const probe = vi.fn().mockRejectedValue(new TypeError("fetch failed"));

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => false,
      settings: SETTINGS,
      keptAt: 0,
    });

    expect(probe).toHaveBeenCalledTimes(10);
  });

  test("stops when cancelled", async () => {
    const probe = vi.fn().mockRejectedValue(CLOSED);

    await waitUntilAnsweredAsOpen({
      probe,
      isCancelled: () => probe.mock.calls.length >= 1,
      settings: SETTINGS,
      keptAt: 0,
    });

    expect(probe).toHaveBeenCalledTimes(1);
  });
});

describe("getPageView", () => {
  const base = {
    isKept: false,
    isTooLate: false,
    isLoading: false,
    isError: false,
  };

  test("a scheduled account can be kept, whatever the device clock says", () => {
    expect(getPageView({ ...base, status: "scheduled" })).toBe("closed");
  });

  test("an account kept on this page is open; any other open one leaves", () => {
    expect(getPageView({ ...base, isKept: true, status: "scheduled" })).toBe(
      "open",
    );
    expect(getPageView({ ...base, status: "open" })).toBe("leaving");
  });

  test("the server's too_late status is too late", () => {
    expect(getPageView({ ...base, status: "too_late" })).toBe("too_late");
  });

  test("a 409 on Keep is too late", () => {
    expect(getPageView({ ...base, isTooLate: true, status: "scheduled" })).toBe(
      "too_late",
    );
  });
});
