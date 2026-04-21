import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePushNotifications } from "./usePushNotifications";

const mockUser = { id: "user-1", email: "test@test.com" };

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: vi.fn(() => ({ user: mockUser })),
}));

const mockIsPushSupported = vi.fn(() => true);
const mockRegisterServiceWorker = vi.fn();
const mockSubscribeToPush = vi.fn();

vi.mock("./registration", () => ({
  isPushSupported: () => mockIsPushSupported(),
  registerServiceWorker: () => mockRegisterServiceWorker(),
  subscribeToPush: (...args: [unknown, unknown]) =>
    mockSubscribeToPush(...args),
}));

const mockFetchVapidPublicKey = vi.fn();
const mockSendSubscriptionToServer = vi.fn();

vi.mock("./api", () => ({
  fetchVapidPublicKey: () => mockFetchVapidPublicKey(),
  sendSubscriptionToServer: (sub: unknown) => mockSendSubscriptionToServer(sub),
}));

describe("usePushNotifications", () => {
  const mockRegistration = {
    pushManager: { getSubscription: vi.fn() },
  };
  const mockSubscription = { endpoint: "https://push.example.com/sub/1" };

  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "granted" },
      configurable: true,
      writable: true,
    });
    Object.defineProperty(navigator, "serviceWorker", {
      value: {
        register: vi.fn(),
        ready: Promise.resolve(mockRegistration),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      },
      configurable: true,
    });
    mockIsPushSupported.mockReturnValue(true);
    mockRegisterServiceWorker.mockResolvedValue(mockRegistration);
    mockSubscribeToPush.mockResolvedValue(mockSubscription);
    mockFetchVapidPublicKey.mockResolvedValue("vapid-key-123");
    mockSendSubscriptionToServer.mockResolvedValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("registers push subscription when all conditions are met", async () => {
    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockSendSubscriptionToServer).toHaveBeenCalledWith(
        mockSubscription,
      );
    });
  });

  it("skips when push is not supported", async () => {
    mockIsPushSupported.mockReturnValue(false);

    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockRegisterServiceWorker).not.toHaveBeenCalled();
    });
  });

  it("skips when Notification permission is not granted", async () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default" },
      configurable: true,
      writable: true,
    });

    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockRegisterServiceWorker).not.toHaveBeenCalled();
    });
  });

  it("skips when no user is authenticated", async () => {
    const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
    vi.mocked(useSupabase).mockReturnValue({
      user: null,
    } as ReturnType<typeof useSupabase>);

    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockIsPushSupported).not.toHaveBeenCalled();
    });
  });

  it("skips when service worker registration fails", async () => {
    mockRegisterServiceWorker.mockResolvedValue(null);

    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockSubscribeToPush).not.toHaveBeenCalled();
    });
  });

  it("skips when VAPID key is not available", async () => {
    mockFetchVapidPublicKey.mockResolvedValue(null);
    delete process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;

    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockSubscribeToPush).not.toHaveBeenCalled();
    });
  });

  it("skips when push subscription fails", async () => {
    mockSubscribeToPush.mockResolvedValue(null);

    renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockSendSubscriptionToServer).not.toHaveBeenCalled();
    });
  });

  it("listens for PUSH_SUBSCRIPTION_CHANGED messages", () => {
    renderHook(() => usePushNotifications());

    expect(navigator.serviceWorker.addEventListener).toHaveBeenCalledWith(
      "message",
      expect.any(Function),
    );
  });
});
