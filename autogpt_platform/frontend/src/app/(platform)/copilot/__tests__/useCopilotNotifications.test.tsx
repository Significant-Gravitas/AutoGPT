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

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
}

function setFocused(focused: boolean) {
  vi.spyOn(document, "hasFocus").mockReturnValue(focused);
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
  vi.restoreAllMocks();
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
    setVisibility("visible");
    setFocused(true);
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

    setVisibility("visible");
    setFocused(true);
    rerender({ active: "sess-1" });
    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
    });
    rerender({ active: null });
    setVisibility("hidden");

    act(() => capturedHandler!(completion));
    await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(2));
  });
});

describe("useCopilotNotifications — background tabs", () => {
  it("keeps the open chat on the finished list until its tab is visible", async () => {
    useCopilotUIStore.setState({ completedSessionIDs: new Set(["sess-1"]) });

    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await act(async () => {});

    expect(useCopilotUIStore.getState().completedSessionIDs).toContain(
      "sess-1",
    );

    setVisibility("visible");
    setFocused(true);
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
    });
  });
});

describe("useCopilotNotifications — unfocused windows", () => {
  it("keeps the open chat on the finished list until its window has focus", async () => {
    setVisibility("visible");
    setFocused(false);
    useCopilotUIStore.setState({ completedSessionIDs: new Set(["sess-1"]) });

    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await act(async () => {});

    expect(useCopilotUIStore.getState().completedSessionIDs).toContain(
      "sess-1",
    );

    setFocused(true);
    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
    });
  });
});

describe("useCopilotNotifications — focus inside an iframe", () => {
  it("does not notify for the open chat while the user works in an embedded frame", async () => {
    setVisibility("visible");
    // Focus moving into an iframe blurs the window but keeps document focus.
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    act(() => {
      window.dispatchEvent(new Event("blur"));
    });

    act(() => {
      capturedHandler!({
        type: "copilot_completion",
        event: "session_completed",
        session_id: "sess-1",
      });
    });
    await act(async () => {});

    expect(NotificationCtor).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().completedSessionIDs).not.toContain(
      "sess-1",
    );
  });
});

describe("useCopilotNotifications — returning into an embedded frame", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("notices the user is back even when no focus event reaches the page", async () => {
    vi.useFakeTimers();
    setVisibility("visible");
    setFocused(false);
    useCopilotUIStore.setState({ completedSessionIDs: new Set(["sess-1"]) });
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });

    // Focus returns straight into an iframe: the document has focus again,
    // but neither "focus" nor "visibilitychange" fires here.
    setFocused(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
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
