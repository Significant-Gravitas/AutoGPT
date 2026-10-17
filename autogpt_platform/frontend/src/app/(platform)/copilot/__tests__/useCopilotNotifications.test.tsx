import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, cleanup, act, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import { COPILOT_COMPLETION_NOTIFICATION } from "../helpers";

type WSHandler = (notification: unknown) => void;
let capturedHandler: WSHandler | null = null;

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketMessage: (_event: string, handler: WSHandler) => {
      capturedHandler = handler;
      return () => {};
    },
  }),
  BackendAPIProvider: ({ children }: { children: ReactNode }) => children,
}));

import { useCopilotNotifications } from "../useCopilotNotifications";
import { useCopilotUIStore } from "../store";

const NotificationCtor = vi.fn();
class FakeNotification {
  onclick: (() => void) | null = null;
  close = vi.fn();
  constructor(title: string, opts: { body: string; icon: string }) {
    NotificationCtor(title, opts);
  }
}

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

beforeEach(() => {
  capturedHandler = null;
  NotificationCtor.mockClear();

  (
    FakeNotification as unknown as { permission: NotificationPermission }
  ).permission = "granted";
  vi.stubGlobal("Notification", FakeNotification);

  useCopilotUIStore.setState({
    completedSessionIDs: new Set(),
    isNotificationsEnabled: true,
    isSoundEnabled: false,
  });

  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => "hidden",
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useCopilotNotifications — OS notification dispatch", () => {
  it("fires a browser notification with the AutoGPT/Task completed copy and 192px icon when a non-active session completes off-screen", async () => {
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });
    expect(capturedHandler).not.toBeNull();

    act(() => {
      capturedHandler!({
        type: "copilot_completion",
        event: "session_completed",
        session_id: "sess-1",
      });
    });

    await waitFor(() => {
      expect(NotificationCtor).toHaveBeenCalledTimes(1);
    });
    expect(NotificationCtor).toHaveBeenCalledWith(
      COPILOT_COMPLETION_NOTIFICATION.title,
      {
        body: COPILOT_COMPLETION_NOTIFICATION.body,
        icon: COPILOT_COMPLETION_NOTIFICATION.icon,
      },
    );
  });

  it("does not fire a notification for unrelated event types", async () => {
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });
    expect(capturedHandler).not.toBeNull();

    act(() => {
      capturedHandler!({
        type: "copilot_completion",
        event: "something_else",
        session_id: "sess-2",
      });
    });

    await waitFor(() => {
      expect(NotificationCtor).not.toHaveBeenCalled();
    });
  });
});

describe("useCopilotNotifications — opening a finished chat", () => {
  it("takes the open chat off the finished list", async () => {
    useCopilotUIStore.setState({
      completedSessionIDs: new Set(["sess-1", "sess-2"]),
    });

    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs).toEqual(
        new Set(["sess-2"]),
      );
    });
  });

  it("notifies again for a chat's next finished turn once it has been opened", async () => {
    const { rerender } = renderHook(
      ({ active }: { active: string | null }) =>
        useCopilotNotifications(active),
      {
        wrapper: makeWrapper(),
        initialProps: { active: null as string | null },
      },
    );
    const completion = {
      type: "copilot_completion",
      event: "session_completed",
      session_id: "sess-1",
    };

    act(() => capturedHandler!(completion));
    await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(1));

    rerender({ active: "sess-1" });
    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
    });
    rerender({ active: null });

    act(() => capturedHandler!(completion));
    await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(2));
  });
});

describe("useCopilotNotifications — tab title", () => {
  it("leaves the tab title alone when sessions finish", async () => {
    document.title = "AutoPilot";
    useCopilotUIStore.setState({
      completedSessionIDs: new Set(["sess-old-1", "sess-old-2"]),
    });
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });

    act(() => {
      capturedHandler!({
        type: "copilot_completion",
        event: "session_completed",
        session_id: "sess-3",
      });
    });

    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs).toContain(
        "sess-3",
      );
    });
    expect(document.title).toBe("AutoPilot");
  });
});
