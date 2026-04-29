import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePushNotifications } from "../usePushNotifications";

const mockUser = { id: "user-1", email: "test@test.com" };

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: vi.fn(() => ({ user: mockUser })),
}));

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

describe("usePushNotifications", () => {
  const mockRegistration = {
    pushManager: {
      getSubscription: vi.fn(),
    },
  };
  const mockSubscription = { endpoint: "https://push.example.com/sub/1" };

  beforeEach(async () => {
    vi.clearAllMocks();
    const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
    vi.mocked(useSupabase).mockReturnValue({
      user: mockUser,
    } as ReturnType<typeof useSupabase>);
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "granted" },
      configurable: true,
      writable: true,
    });
    Object.defineProperty(navigator, "serviceWorker", {
      value: {
        register: vi.fn(),
        getRegistration: vi.fn().mockResolvedValue(mockRegistration),
        ready: Promise.resolve(mockRegistration),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
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
    mockRemoveSubscriptionFromServer.mockResolvedValue(true);
    mockUnsubscribeFromPush.mockResolvedValue(true);
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

  it("unsubscribes on logout after having been authenticated", async () => {
    const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
    const { rerender } = renderHook(() => usePushNotifications());

    await waitFor(() => {
      expect(mockSendSubscriptionToServer).toHaveBeenCalled();
    });

    vi.mocked(useSupabase).mockReturnValue({
      user: null,
    } as ReturnType<typeof useSupabase>);
    rerender();

    await waitFor(() => {
      expect(mockRemoveSubscriptionFromServer).toHaveBeenCalledWith(
        mockSubscription.endpoint,
      );
    });
    expect(mockUnsubscribeFromPush).toHaveBeenCalledWith(mockRegistration);
  });

  it("does not call teardown when user is null from the start", async () => {
    const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
    vi.mocked(useSupabase).mockReturnValue({
      user: null,
    } as ReturnType<typeof useSupabase>);

    renderHook(() => usePushNotifications());

    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(mockRemoveSubscriptionFromServer).not.toHaveBeenCalled();
    expect(mockUnsubscribeFromPush).not.toHaveBeenCalled();
  });
});
