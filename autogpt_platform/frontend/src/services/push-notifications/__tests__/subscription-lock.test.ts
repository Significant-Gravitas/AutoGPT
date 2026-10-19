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
    const request = vi.fn((_name: string, action: () => Promise<string>) =>
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
      expect.any(Function),
    );
  });

  it("runs the action directly where Web Locks are missing", async () => {
    await expect(withPushSubscriptionLock(async () => "done")).resolves.toBe(
      "done",
    );
  });
});
