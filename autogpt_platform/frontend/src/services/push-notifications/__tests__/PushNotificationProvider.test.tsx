import { render, waitFor } from "@/tests/integrations/test-utils";
import {
  afterAll,
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { PushNotificationProvider } from "../PushNotificationProvider";

const { pathnameMock, searchParamsMock } = vi.hoisted(() => ({
  pathnameMock: vi.fn(() => "/copilot"),
  searchParamsMock: vi.fn(() => new URLSearchParams("sessionId=A")),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => pathnameMock(),
  useSearchParams: () => searchParamsMock(),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  useParams: () => ({}),
}));

const mockUser = { id: "user-1", email: "test@test.com" };
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: vi.fn(() => ({ user: mockUser })),
}));

const { notificationsEnabledMock } = vi.hoisted(() => ({
  notificationsEnabledMock: vi.fn(() => true),
}));
vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: (
    selector: (s: { isNotificationsEnabled: boolean }) => unknown,
  ) => selector({ isNotificationsEnabled: notificationsEnabledMock() }),
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

const mockSubscription = { endpoint: "https://push.example.com/sub/1" };
const mockRegistration = {
  pushManager: {
    getSubscription: vi.fn(),
  },
};

const mockPostMessage = vi.fn();
const swListeners: Record<string, Array<(e?: unknown) => void>> = {};
const swMessageListeners: Array<(e: MessageEvent) => void> = [];

function installServiceWorker() {
  Object.defineProperty(navigator, "serviceWorker", {
    value: {
      controller: { postMessage: mockPostMessage },
      register: vi.fn(),
      getRegistration: vi.fn().mockResolvedValue(mockRegistration),
      ready: Promise.resolve(mockRegistration),
      addEventListener: (name: string, fn: (e?: unknown) => void) => {
        if (name === "message") {
          swMessageListeners.push(fn as (e: MessageEvent) => void);
          return;
        }
        if (!swListeners[name]) swListeners[name] = [];
        swListeners[name].push(fn);
      },
      removeEventListener: (name: string, fn: (e?: unknown) => void) => {
        if (name === "message") {
          const idx = swMessageListeners.indexOf(
            fn as (e: MessageEvent) => void,
          );
          if (idx !== -1) swMessageListeners.splice(idx, 1);
          return;
        }
        const list = swListeners[name];
        if (!list) return;
        const idx = list.indexOf(fn);
        if (idx !== -1) list.splice(idx, 1);
      },
    },
    configurable: true,
    writable: true,
  });
}

function uninstallServiceWorker() {
  // Fully remove the property so both `"serviceWorker" in navigator` and
  // truthy checks short-circuit, matching the "API unavailable" branch.
  delete (navigator as unknown as { serviceWorker?: unknown }).serviceWorker;
}

const originalNotificationDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  "Notification",
);
const originalServiceWorkerDescriptor = Object.getOwnPropertyDescriptor(
  navigator,
  "serviceWorker",
);
const originalVapidPublicKey = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;

function restoreProperty(
  target: object,
  key: string,
  descriptor: PropertyDescriptor | undefined,
) {
  if (descriptor) {
    Object.defineProperty(target, key, descriptor);
    return;
  }
  delete (target as Record<string, unknown>)[key];
}

describe("PushNotificationProvider", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    Object.keys(swListeners).forEach((k) => delete swListeners[k]);
    swMessageListeners.length = 0;

    pathnameMock.mockReturnValue("/copilot");
    searchParamsMock.mockReturnValue(new URLSearchParams("sessionId=A"));
    notificationsEnabledMock.mockReturnValue(true);

    const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
    vi.mocked(useSupabase).mockReturnValue({
      user: mockUser,
    } as ReturnType<typeof useSupabase>);

    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "granted" },
      configurable: true,
      writable: true,
    });

    installServiceWorker();
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

  afterAll(() => {
    // Tests in this file mutate three module-scoped globals — `Notification`,
    // `navigator.serviceWorker`, and `process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY`.
    // beforeEach re-stubs them between tests inside this file, but without an
    // afterAll the mutations leak to any other test file scheduled into the
    // same Vitest worker.
    restoreProperty(globalThis, "Notification", originalNotificationDescriptor);
    restoreProperty(
      navigator,
      "serviceWorker",
      originalServiceWorkerDescriptor,
    );
    if (originalVapidPublicKey === undefined) {
      delete process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;
    } else {
      process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY = originalVapidPublicKey;
    }
  });

  describe("push subscription registration", () => {
    it("registers push subscription when all conditions are met", async () => {
      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockSendSubscriptionToServer).toHaveBeenCalledWith(
          mockSubscription,
        );
      });
    });

    // Skip-path tests use `mockPostMessage` as a deterministic "mount effects
    // flushed" anchor before asserting that the push-registration helpers were
    // NOT called. The URL reporter posts on mount regardless of push-support /
    // permission / auth / VAPID state, so it gives a reliable signal that the
    // provider's effects have run — without it, `expect(...).not.toHaveBeenCalled()`
    // can pass before the effect that would have called the mock fires.
    it("skips when push is not supported", async () => {
      mockIsPushSupported.mockReturnValue(false);

      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockPostMessage).toHaveBeenCalled();
      });
      expect(mockRegisterServiceWorker).not.toHaveBeenCalled();
    });

    it("skips when Notification permission is not granted", async () => {
      Object.defineProperty(globalThis, "Notification", {
        value: { permission: "default" },
        configurable: true,
        writable: true,
      });

      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockPostMessage).toHaveBeenCalled();
      });
      expect(mockRegisterServiceWorker).not.toHaveBeenCalled();
    });

    it("skips when no user is authenticated", async () => {
      const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
      vi.mocked(useSupabase).mockReturnValue({
        user: null,
      } as ReturnType<typeof useSupabase>);

      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockPostMessage).toHaveBeenCalled();
      });
      expect(mockIsPushSupported).not.toHaveBeenCalled();
    });

    it("skips when service worker registration fails", async () => {
      mockRegisterServiceWorker.mockResolvedValue(null);

      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockRegisterServiceWorker).toHaveBeenCalled();
      });
      expect(mockSubscribeToPush).not.toHaveBeenCalled();
    });

    it("skips when VAPID key is not available", async () => {
      mockFetchVapidPublicKey.mockResolvedValue(null);
      delete process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;

      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockFetchVapidPublicKey).toHaveBeenCalled();
      });
      expect(mockSubscribeToPush).not.toHaveBeenCalled();
    });

    it("skips when push subscription fails", async () => {
      mockSubscribeToPush.mockResolvedValue(null);

      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockSubscribeToPush).toHaveBeenCalled();
      });
      expect(mockSendSubscriptionToServer).not.toHaveBeenCalled();
    });

    it("renews the subscription when PUSH_SUBSCRIPTION_CHANGED arrives", async () => {
      render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockSendSubscriptionToServer).toHaveBeenCalledWith(
          mockSubscription,
        );
      });
      expect(swMessageListeners.length).toBeGreaterThan(0);

      mockSubscribeToPush.mockClear();
      mockSendSubscriptionToServer.mockClear();

      swMessageListeners.forEach((fn) =>
        fn(
          new MessageEvent("message", {
            data: { type: "PUSH_SUBSCRIPTION_CHANGED" },
          }),
        ),
      );

      await waitFor(() => {
        expect(mockSubscribeToPush).toHaveBeenCalled();
      });
      expect(mockSendSubscriptionToServer).toHaveBeenCalledWith(
        mockSubscription,
      );
    });

    it("unsubscribes on logout after having been authenticated", async () => {
      const { useSupabase } = await import("@/lib/supabase/hooks/useSupabase");
      const { rerender } = render(<PushNotificationProvider />);

      await waitFor(() => {
        expect(mockSendSubscriptionToServer).toHaveBeenCalled();
      });

      vi.mocked(useSupabase).mockReturnValue({
        user: null,
      } as ReturnType<typeof useSupabase>);
      rerender(<PushNotificationProvider />);

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

      render(<PushNotificationProvider />);

      // Wait for the provider's mount-time side effects (URL reporter + SW
      // listener registration) to flush before asserting absence of teardown.
      await waitFor(() => {
        expect(mockPostMessage).toHaveBeenCalled();
      });
      expect(mockRemoveSubscriptionFromServer).not.toHaveBeenCalled();
      expect(mockUnsubscribeFromPush).not.toHaveBeenCalled();
    });
  });

  describe("client URL reporting", () => {
    it("posts the current URL on mount", () => {
      render(<PushNotificationProvider />);

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "CLIENT_URL",
        url: "/copilot?sessionId=A",
      });
    });

    it("includes the pathname only when there are no query params", () => {
      searchParamsMock.mockReturnValue(new URLSearchParams());

      render(<PushNotificationProvider />);

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "CLIENT_URL",
        url: "/copilot",
      });
    });

    it("re-posts when pathname changes", () => {
      const { rerender } = render(<PushNotificationProvider />);
      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "CLIENT_URL",
        url: "/copilot?sessionId=A",
      });

      pathnameMock.mockReturnValue("/library");
      searchParamsMock.mockReturnValue(new URLSearchParams());
      rerender(<PushNotificationProvider />);

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "CLIENT_URL",
        url: "/library",
      });
    });

    it("re-posts the URL when the SW controller changes", async () => {
      render(<PushNotificationProvider />);
      mockPostMessage.mockClear();

      swListeners.controllerchange?.forEach((fn) => fn());

      await waitFor(() => {
        expect(mockPostMessage).toHaveBeenCalledWith({
          type: "CLIENT_URL",
          url: "/copilot?sessionId=A",
        });
      });
    });
  });

  describe("notifications-enabled reporting", () => {
    it("posts the toggle value on mount when enabled", () => {
      notificationsEnabledMock.mockReturnValue(true);

      render(<PushNotificationProvider />);

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "NOTIFICATIONS_ENABLED",
        value: true,
      });
    });

    it("posts the toggle value on mount when disabled", () => {
      notificationsEnabledMock.mockReturnValue(false);

      render(<PushNotificationProvider />);

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "NOTIFICATIONS_ENABLED",
        value: false,
      });
    });

    it("re-posts when the toggle flips", () => {
      notificationsEnabledMock.mockReturnValue(true);
      const { rerender } = render(<PushNotificationProvider />);
      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "NOTIFICATIONS_ENABLED",
        value: true,
      });

      notificationsEnabledMock.mockReturnValue(false);
      rerender(<PushNotificationProvider />);

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "NOTIFICATIONS_ENABLED",
        value: false,
      });
    });

    it("re-posts the toggle on SW controllerchange", () => {
      notificationsEnabledMock.mockReturnValue(false);
      render(<PushNotificationProvider />);
      mockPostMessage.mockClear();

      swListeners.controllerchange?.forEach((fn) => fn());

      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "NOTIFICATIONS_ENABLED",
        value: false,
      });
    });
  });

  describe("when serviceWorker API is unavailable", () => {
    it("does not post any URL or toggle messages", () => {
      uninstallServiceWorker();

      render(<PushNotificationProvider />);

      expect(mockPostMessage).not.toHaveBeenCalled();
    });
  });
});
