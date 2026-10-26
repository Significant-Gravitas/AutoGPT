import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setupPushSubscription,
  teardownPushSubscription,
  teardownPushSubscriptionBeforeLogout,
} from "../helpers";

const mockIsPushSupported = vi.fn(() => true);
const mockRegisterServiceWorker = vi.fn();
const mockSubscribeToPush = vi.fn();
const mockUnsubscribeFromPush = vi.fn();

vi.mock("../registration", () => ({
  isPushSupported: () => mockIsPushSupported(),
  registerServiceWorker: () => mockRegisterServiceWorker(),
  subscribeToPush: (...args: [unknown, unknown]) =>
    mockSubscribeToPush(...args),
  unsubscribeFromPush: (sub: unknown) => mockUnsubscribeFromPush(sub),
}));

const mockFetchVapidPublicKey = vi.fn();
const mockSendSubscriptionToServer = vi.fn();
const mockRemoveSubscriptionFromServer = vi.fn();

vi.mock("../api", () => ({
  fetchVapidPublicKey: () => mockFetchVapidPublicKey(),
  sendSubscriptionToServer: (sub: unknown) => mockSendSubscriptionToServer(sub),
  removeSubscriptionFromServer: (endpoint: string) =>
    mockRemoveSubscriptionFromServer(endpoint),
}));

const mockSubscription = { endpoint: "https://push.example.com/sub/1" };
const mockRegistration = {
  pushManager: { getSubscription: vi.fn() },
};

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(globalThis, "Notification", {
    value: { permission: "granted" },
    configurable: true,
    writable: true,
  });
  Object.defineProperty(navigator, "serviceWorker", {
    value: {
      ready: Promise.resolve(mockRegistration),
      getRegistration: vi.fn().mockResolvedValue(mockRegistration),
    },
    configurable: true,
  });
  mockRegistration.pushManager.getSubscription = vi
    .fn()
    .mockResolvedValue(mockSubscription);
  mockIsPushSupported.mockReturnValue(true);
  mockRegisterServiceWorker.mockResolvedValue(mockRegistration);
  mockSubscribeToPush.mockResolvedValue(mockSubscription);
  mockFetchVapidPublicKey.mockResolvedValue("vapid-key-123");
  mockSendSubscriptionToServer.mockResolvedValue(true);
  mockUnsubscribeFromPush.mockResolvedValue(true);
  mockRemoveSubscriptionFromServer.mockResolvedValue(true);
  delete process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("setupPushSubscription", () => {
  it("returns true on the full happy path", async () => {
    const result = await setupPushSubscription();
    expect(result).toBe(true);
    expect(mockSendSubscriptionToServer).toHaveBeenCalledWith(mockSubscription);
  });

  it("returns false when push is not supported", async () => {
    mockIsPushSupported.mockReturnValue(false);
    const result = await setupPushSubscription();
    expect(result).toBe(false);
    expect(mockRegisterServiceWorker).not.toHaveBeenCalled();
  });

  it("returns false when notification permission is not granted", async () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default" },
      configurable: true,
      writable: true,
    });
    const result = await setupPushSubscription();
    expect(result).toBe(false);
    expect(mockRegisterServiceWorker).not.toHaveBeenCalled();
  });

  it("returns false when service worker registration fails", async () => {
    mockRegisterServiceWorker.mockResolvedValue(null);
    const result = await setupPushSubscription();
    expect(result).toBe(false);
    expect(mockSubscribeToPush).not.toHaveBeenCalled();
  });

  it("returns false when no vapid key is available", async () => {
    mockFetchVapidPublicKey.mockResolvedValue(null);
    const result = await setupPushSubscription();
    expect(result).toBe(false);
    expect(mockSubscribeToPush).not.toHaveBeenCalled();
  });

  it("uses NEXT_PUBLIC_VAPID_PUBLIC_KEY when set, skipping the server fetch", async () => {
    process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY = "build-time-key";
    await setupPushSubscription();
    expect(mockFetchVapidPublicKey).not.toHaveBeenCalled();
    expect(mockSubscribeToPush).toHaveBeenCalledWith(
      mockRegistration,
      "build-time-key",
    );
  });

  it("returns false when subscribeToPush fails", async () => {
    mockSubscribeToPush.mockResolvedValue(null);
    const result = await setupPushSubscription();
    expect(result).toBe(false);
    expect(mockSendSubscriptionToServer).not.toHaveBeenCalled();
  });
});

describe("teardownPushSubscription", () => {
  it("removes server subscription and unsubscribes locally", async () => {
    await teardownPushSubscription();
    expect(mockRemoveSubscriptionFromServer).toHaveBeenCalledWith(
      mockSubscription.endpoint,
    );
    expect(mockUnsubscribeFromPush).toHaveBeenCalledWith(mockSubscription);
  });

  it("no-ops when serviceWorker is unavailable", async () => {
    Object.defineProperty(navigator, "serviceWorker", {
      value: undefined,
      configurable: true,
    });
    await teardownPushSubscription();
    expect(mockRemoveSubscriptionFromServer).not.toHaveBeenCalled();
    expect(mockUnsubscribeFromPush).not.toHaveBeenCalled();
  });

  it("no-ops when no registration exists", async () => {
    Object.defineProperty(navigator, "serviceWorker", {
      value: {
        getRegistration: vi.fn().mockResolvedValue(null),
      },
      configurable: true,
    });
    await teardownPushSubscription();
    expect(mockRemoveSubscriptionFromServer).not.toHaveBeenCalled();
    expect(mockUnsubscribeFromPush).not.toHaveBeenCalled();
  });

  it("no-ops when there is no active subscription", async () => {
    mockRegistration.pushManager.getSubscription = vi
      .fn()
      .mockResolvedValue(null);
    await teardownPushSubscription();
    expect(mockRemoveSubscriptionFromServer).not.toHaveBeenCalled();
    expect(mockUnsubscribeFromPush).not.toHaveBeenCalled();
  });
});

describe("teardownPushSubscriptionBeforeLogout", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("removes the subscription from the server and the browser", async () => {
    await teardownPushSubscriptionBeforeLogout();
    expect(mockRemoveSubscriptionFromServer).toHaveBeenCalledWith(
      mockSubscription.endpoint,
    );
    expect(mockUnsubscribeFromPush).toHaveBeenCalledWith(mockSubscription);
  });

  it("unsubscribes the browser without waiting for the server", async () => {
    mockRemoveSubscriptionFromServer.mockReturnValue(new Promise(() => {}));

    void teardownPushSubscriptionBeforeLogout();

    await vi.waitFor(() =>
      expect(mockUnsubscribeFromPush).toHaveBeenCalledWith(mockSubscription),
    );
  });

  it("never unsubscribes the next account's subscription after timing out", async () => {
    vi.useFakeTimers();
    let answerServer: (ok: boolean) => void = () => {};
    mockRemoveSubscriptionFromServer.mockReturnValue(
      new Promise<boolean>((resolve) => (answerServer = resolve)),
    );

    await Promise.all([
      teardownPushSubscriptionBeforeLogout(),
      vi.advanceTimersByTimeAsync(5_000),
    ]);
    const nextAccountSubscription = {
      endpoint: "https://push.example.com/sub/2",
    };
    mockRegistration.pushManager.getSubscription = vi
      .fn()
      .mockResolvedValue(nextAccountSubscription);
    answerServer(true);
    await vi.advanceTimersByTimeAsync(0);

    expect(mockUnsubscribeFromPush.mock.calls).toEqual([[mockSubscription]]);
  });

  it("does not hold up logout when the server never answers", async () => {
    vi.useFakeTimers();
    mockRemoveSubscriptionFromServer.mockReturnValue(new Promise(() => {}));

    const done = vi.fn();
    void teardownPushSubscriptionBeforeLogout().then(done);
    await vi.advanceTimersByTimeAsync(5_000);

    expect(done).toHaveBeenCalled();
  });

  it("does not throw when tearing down fails", async () => {
    mockRemoveSubscriptionFromServer.mockRejectedValue(new Error("offline"));
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    await expect(
      teardownPushSubscriptionBeforeLogout(),
    ).resolves.toBeUndefined();

    consoleError.mockRestore();
  });
});

describe("push subscription lock across tabs", () => {
  afterEach(() => {
    Object.defineProperty(navigator, "locks", {
      value: undefined,
      configurable: true,
    });
  });

  // A first-come, first-served lock like the browser's, shared by "tabs"
  // that are just concurrent calls here.
  function installSerialLock() {
    let tail: Promise<unknown> = Promise.resolve();
    const request = vi.fn(
      (
        _name: string,
        _options: unknown,
        action: () => Promise<unknown>,
      ): Promise<unknown> => {
        const run = tail.then(action);
        tail = run.catch(() => undefined);
        return run;
      },
    );
    Object.defineProperty(navigator, "locks", {
      value: { request },
      configurable: true,
    });
  }

  it("makes another tab's subscribe wait until this tab has unsubscribed", async () => {
    installSerialLock();
    let finishUnsubscribe: (ok: boolean) => void = () => {};
    mockUnsubscribeFromPush.mockReturnValue(
      new Promise<boolean>((resolve) => (finishUnsubscribe = resolve)),
    );

    const tearingDown = teardownPushSubscription();
    const settingUp = setupPushSubscription();
    await vi.waitFor(() =>
      expect(mockUnsubscribeFromPush).toHaveBeenCalledWith(mockSubscription),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockSubscribeToPush).not.toHaveBeenCalled();

    finishUnsubscribe(true);
    await Promise.all([tearingDown, settingUp]);

    expect(mockSubscribeToPush).toHaveBeenCalledOnce();
  });
});
