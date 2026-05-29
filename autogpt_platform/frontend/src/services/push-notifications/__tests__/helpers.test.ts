import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setupPushSubscription, teardownPushSubscription } from "../helpers";

const mockIsPushSupported = vi.fn(() => true);
const mockRegisterServiceWorker = vi.fn();
const mockSubscribeToPush = vi.fn();
const mockUnsubscribeFromPush = vi.fn();

vi.mock("../registration", () => ({
  isPushSupported: () => mockIsPushSupported(),
  registerServiceWorker: () => mockRegisterServiceWorker(),
  subscribeToPush: (...args: [unknown, unknown]) =>
    mockSubscribeToPush(...args),
  unsubscribeFromPush: (reg: unknown) => mockUnsubscribeFromPush(reg),
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
    expect(mockUnsubscribeFromPush).toHaveBeenCalledWith(mockRegistration);
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
