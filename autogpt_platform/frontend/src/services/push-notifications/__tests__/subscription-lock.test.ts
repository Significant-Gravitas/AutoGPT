import { afterEach, describe, expect, it, vi } from "vitest";
import { withPushSubscriptionLock } from "../subscription-lock";

afterEach(() => {
  Object.defineProperty(navigator, "locks", {
    value: undefined,
    configurable: true,
  });
});

describe("withPushSubscriptionLock", () => {
  it("runs the action under the browser-wide push subscription lock", async () => {
    const request = vi.fn(
      (_name: string, _options: unknown, action: () => Promise<string>) =>
        action(),
    );
    Object.defineProperty(navigator, "locks", {
      value: { request },
      configurable: true,
    });

    await expect(withPushSubscriptionLock(async () => "done")).resolves.toBe(
      "done",
    );
    expect(request).toHaveBeenCalledWith(
      "autogpt-push-subscription",
      expect.anything(),
      expect.any(Function),
    );
  });

  it("runs the action directly where Web Locks are missing", async () => {
    await expect(withPushSubscriptionLock(async () => "done")).resolves.toBe(
      "done",
    );
  });
});

describe("withPushSubscriptionLock with a wait limit", () => {
  afterEach(() => {
    vi.useRealTimers();
    Object.defineProperty(navigator, "locks", {
      value: undefined,
      configurable: true,
    });
  });

  // Another tab holds the lock and never lets go; the request only ends
  // when its signal aborts, as the browser's does.
  function installHeldLock() {
    const request = vi.fn(
      (_name: string, options: { signal?: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );
    Object.defineProperty(navigator, "locks", {
      value: { request },
      configurable: true,
    });
  }

  it("runs the action unlocked once the wait runs out", async () => {
    vi.useFakeTimers();
    installHeldLock();
    const action = vi.fn(async () => "done");

    const running = withPushSubscriptionLock(action, { maxWaitMs: 2_000 });
    await vi.advanceTimersByTimeAsync(1_999);
    expect(action).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    await expect(running).resolves.toBe("done");
    expect(action).toHaveBeenCalledOnce();
  });

  it("runs the action once under the lock when it is granted in time", async () => {
    vi.useFakeTimers();
    const request = vi.fn(
      (_name: string, _options: unknown, action: () => Promise<unknown>) =>
        action(),
    );
    Object.defineProperty(navigator, "locks", {
      value: { request },
      configurable: true,
    });
    const action = vi.fn(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5_000));
      return "done";
    });

    const running = withPushSubscriptionLock(action, { maxWaitMs: 2_000 });
    await vi.advanceTimersByTimeAsync(5_000);

    await expect(running).resolves.toBe("done");
    expect(action).toHaveBeenCalledOnce();
  });
});
