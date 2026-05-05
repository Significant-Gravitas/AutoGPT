import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotNotifications } from "../useCopilotNotifications";

type WsHandler = (notification: Record<string, unknown>) => void;
let capturedHandler: WsHandler | null = null;

const mockApi = {
  onWebSocketMessage: vi.fn((_method: string, handler: WsHandler) => {
    capturedHandler = handler;
    return () => {
      capturedHandler = null;
    };
  }),
};

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => mockApi,
}));

const mockStoreState = {
  completedSessionIDs: new Set<string>(),
  addCompletedSession: vi.fn(),
  isNotificationsEnabled: true,
  isSoundEnabled: false,
};

vi.mock("../store", () => ({
  useCopilotUIStore: {
    getState: () => mockStoreState,
    setState: vi.fn(),
  },
}));

vi.mock("../helpers", () => ({
  ORIGINAL_TITLE: "AutoGPT",
  formatNotificationTitle: (count: number) => `(${count}) AutoPilot is ready`,
  parseSessionIDs: () => new Set<string>(),
}));

vi.mock("@/services/storage/local-storage", () => ({
  Key: { COPILOT_COMPLETED_SESSIONS: "copilot-completed-sessions" },
  storage: { get: vi.fn(), set: vi.fn(), remove: vi.fn() },
}));

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getGetV2ListSessionsQueryKey: () => ["sessions"],
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

const mockNotificationCtor = vi.fn();

function fireCompletion(sessionID: string) {
  if (!capturedHandler) throw new Error("handler not captured");
  capturedHandler({
    type: "copilot_completion",
    event: "session_completed",
    session_id: sessionID,
  });
}

describe("useCopilotNotifications — push SW suppression", () => {
  beforeEach(() => {
    capturedHandler = null;
    mockApi.onWebSocketMessage.mockClear();
    mockStoreState.completedSessionIDs = new Set();
    mockStoreState.isNotificationsEnabled = true;
    mockStoreState.isSoundEnabled = false;
    mockStoreState.addCompletedSession.mockClear();

    mockNotificationCtor.mockClear();
    const NotificationStub = function Notification(
      this: unknown,
      title: string,
      opts: unknown,
    ) {
      mockNotificationCtor(title, opts);
    } as unknown as typeof Notification;
    (NotificationStub as unknown as { permission: string }).permission =
      "granted";
    Object.defineProperty(globalThis, "Notification", {
      value: NotificationStub,
      configurable: true,
      writable: true,
    });

    const AudioStub = function AudioStub(this: Record<string, unknown>) {
      this.volume = 0;
      this.play = () => Promise.resolve();
    } as unknown as typeof Audio;
    vi.stubGlobal("Audio", AudioStub);

    // Start with "user away" so the hook triggers the notification path:
    // blur fires after mount; simulate by setting visibilityState to hidden.
    Object.defineProperty(document, "visibilityState", {
      value: "hidden",
      configurable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function installServiceWorker(args: {
    registration: unknown | null;
    pushSubscription?: unknown | null;
  }) {
    const swReg =
      args.registration === null
        ? null
        : {
            pushManager: {
              getSubscription: vi
                .fn()
                .mockResolvedValue(args.pushSubscription ?? null),
            },
          };
    Object.defineProperty(navigator, "serviceWorker", {
      value: {
        getRegistration: vi.fn().mockResolvedValue(swReg),
      },
      configurable: true,
    });
  }

  it("skips in-page Notification when push SW subscription is active", async () => {
    installServiceWorker({
      registration: {},
      pushSubscription: { endpoint: "https://fcm.googleapis.com/x" },
    });

    renderHook(() => useCopilotNotifications(null));

    await act(async () => {
      fireCompletion("session-1");
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockStoreState.addCompletedSession).toHaveBeenCalledWith(
      "session-1",
    );
    expect(mockNotificationCtor).not.toHaveBeenCalled();
  });

  it("uses in-page Notification when SW registration has no push subscription", async () => {
    installServiceWorker({
      registration: {},
      pushSubscription: null,
    });

    renderHook(() => useCopilotNotifications(null));

    await act(async () => {
      fireCompletion("session-2");
    });
    await waitFor(() => {
      expect(mockNotificationCtor).toHaveBeenCalledTimes(1);
    });
    expect(mockNotificationCtor.mock.calls[0][0]).toBe("AutoPilot is ready");
  });

  it("uses in-page Notification when no SW registration exists", async () => {
    installServiceWorker({ registration: null });

    renderHook(() => useCopilotNotifications(null));

    await act(async () => {
      fireCompletion("session-3");
    });
    await waitFor(() => {
      expect(mockNotificationCtor).toHaveBeenCalledTimes(1);
    });
  });

  it("uses in-page Notification when serviceWorker API is unavailable", async () => {
    Object.defineProperty(navigator, "serviceWorker", {
      value: undefined,
      configurable: true,
    });

    renderHook(() => useCopilotNotifications(null));

    await act(async () => {
      fireCompletion("session-4");
    });
    await waitFor(() => {
      expect(mockNotificationCtor).toHaveBeenCalledTimes(1);
    });
  });
});
