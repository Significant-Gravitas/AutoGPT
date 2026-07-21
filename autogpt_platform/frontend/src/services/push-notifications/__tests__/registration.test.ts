import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isPushSupported,
  registerServiceWorker,
  subscribeToPush,
  unsubscribeFromPush,
} from "../registration";

describe("isPushSupported", () => {
  const savedNavigator = globalThis.navigator;

  afterEach(() => {
    Object.defineProperty(globalThis, "navigator", {
      value: savedNavigator,
      configurable: true,
    });
  });

  it("returns true when all browser APIs exist", () => {
    Object.defineProperty(globalThis.navigator, "serviceWorker", {
      value: {},
      configurable: true,
    });
    Object.defineProperty(window, "PushManager", {
      value: class {},
      configurable: true,
    });
    Object.defineProperty(window, "Notification", {
      value: class {},
      configurable: true,
    });

    expect(isPushSupported()).toBe(true);
  });

  it("returns false when serviceWorker is missing", () => {
    const nav = {} as Navigator;
    Object.defineProperty(globalThis, "navigator", {
      value: nav,
      configurable: true,
    });
    Object.defineProperty(window, "PushManager", {
      value: class {},
      configurable: true,
    });
    Object.defineProperty(window, "Notification", {
      value: class {},
      configurable: true,
    });

    expect(isPushSupported()).toBe(false);
  });

  it("returns false when PushManager is missing", () => {
    Object.defineProperty(globalThis.navigator, "serviceWorker", {
      value: {},
      configurable: true,
    });
    // @ts-expect-error — intentionally removing PushManager
    delete window.PushManager;
    Object.defineProperty(window, "Notification", {
      value: class {},
      configurable: true,
    });

    expect(isPushSupported()).toBe(false);
  });

  it("returns false when Notification is missing", () => {
    Object.defineProperty(globalThis.navigator, "serviceWorker", {
      value: {},
      configurable: true,
    });
    Object.defineProperty(window, "PushManager", {
      value: class {},
      configurable: true,
    });
    // @ts-expect-error — intentionally removing Notification
    delete window.Notification;

    expect(isPushSupported()).toBe(false);
  });
});

describe("registerServiceWorker", () => {
  it("registers a service worker at /push-sw.js", async () => {
    const mockRegistration = { scope: "/" } as ServiceWorkerRegistration;
    Object.defineProperty(globalThis.navigator, "serviceWorker", {
      value: {
        register: vi.fn().mockResolvedValue(mockRegistration),
      },
      configurable: true,
    });

    const result = await registerServiceWorker();

    expect(navigator.serviceWorker.register).toHaveBeenCalledWith(
      "/push-sw.js",
      { scope: "/" },
    );
    expect(result).toBe(mockRegistration);
  });

  it("returns null when registration fails", async () => {
    Object.defineProperty(globalThis.navigator, "serviceWorker", {
      value: {
        register: vi.fn().mockRejectedValue(new Error("SW failed")),
      },
      configurable: true,
    });

    const result = await registerServiceWorker();

    expect(result).toBeNull();
  });

  it("returns null when serviceWorker is not in navigator", async () => {
    const nav = {} as Navigator;
    const original = globalThis.navigator;
    Object.defineProperty(globalThis, "navigator", {
      value: nav,
      configurable: true,
    });

    const result = await registerServiceWorker();

    expect(result).toBeNull();

    Object.defineProperty(globalThis, "navigator", {
      value: original,
      configurable: true,
    });
  });
});

describe("subscribeToPush", () => {
  it("returns existing subscription if one exists", async () => {
    const existingSub = { endpoint: "https://push.example.com" };
    const registration = {
      pushManager: {
        getSubscription: vi.fn().mockResolvedValue(existingSub),
        subscribe: vi.fn(),
      },
    } as unknown as ServiceWorkerRegistration;

    const result = await subscribeToPush(registration, "fake-vapid-key");

    expect(result).toBe(existingSub);
    expect(registration.pushManager.subscribe).not.toHaveBeenCalled();
  });

  it("creates new subscription when none exists", async () => {
    const newSub = { endpoint: "https://push.example.com/new" };
    const registration = {
      pushManager: {
        getSubscription: vi.fn().mockResolvedValue(null),
        subscribe: vi.fn().mockResolvedValue(newSub),
      },
    } as unknown as ServiceWorkerRegistration;

    // Valid base64url — decodes to 4 bytes (AAAA in base64url = 3 zero bytes)
    const vapidKey = "AAAA";

    const result = await subscribeToPush(registration, vapidKey);

    expect(result).toBe(newSub);
    expect(registration.pushManager.subscribe).toHaveBeenCalledWith({
      userVisibleOnly: true,
      applicationServerKey: expect.any(ArrayBuffer),
    });
  });

  it("returns null on subscribe failure", async () => {
    const registration = {
      pushManager: {
        getSubscription: vi.fn().mockResolvedValue(null),
        subscribe: vi.fn().mockRejectedValue(new Error("denied")),
      },
    } as unknown as ServiceWorkerRegistration;

    const result = await subscribeToPush(registration, "fake-vapid-key");

    expect(result).toBeNull();
  });
});

describe("unsubscribeFromPush", () => {
  it("unsubscribes existing subscription", async () => {
    const subscription = { unsubscribe: vi.fn().mockResolvedValue(true) };
    const registration = {
      pushManager: {
        getSubscription: vi.fn().mockResolvedValue(subscription),
      },
    } as unknown as ServiceWorkerRegistration;

    const result = await unsubscribeFromPush(registration);

    expect(result).toBe(true);
    expect(subscription.unsubscribe).toHaveBeenCalled();
  });

  it("returns true when no subscription exists", async () => {
    const registration = {
      pushManager: {
        getSubscription: vi.fn().mockResolvedValue(null),
      },
    } as unknown as ServiceWorkerRegistration;

    const result = await unsubscribeFromPush(registration);

    expect(result).toBe(true);
  });

  it("returns false when unsubscribe fails", async () => {
    const subscription = {
      unsubscribe: vi.fn().mockRejectedValue(new Error("failed")),
    };
    const registration = {
      pushManager: {
        getSubscription: vi.fn().mockResolvedValue(subscription),
      },
    } as unknown as ServiceWorkerRegistration;

    const result = await unsubscribeFromPush(registration);

    expect(result).toBe(false);
  });
});
