import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, cleanup, act, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

import { getPostV2MarkSessionReadMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { COPILOT_COMPLETION_NOTIFICATION } from "../helpers";
import { SESSION_GROUPS_QUERY_KEY } from "../sessionGroups";

type WSHandler = (notification: unknown) => void;
let capturedHandler: WSHandler | null = null;

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketMessage: (_event: string, handler: WSHandler) => {
      capturedHandler = handler;
      return () => {};
    },
    onWebSocketReconnect: () => () => {},
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

let lastQueryClient: QueryClient;

// The read has settled, list re-read included, not just reached the server:
// until then the hook ignores the lists, so a cache update would be missed.
async function waitForFirstRead() {
  await waitFor(() => expect(markedRead).toEqual(["sess-1"]));
  await waitFor(() => expect(lastQueryClient.isMutating()).toBe(0));
}

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  lastQueryClient = queryClient;
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

let markedRead: string[] = [];

beforeEach(() => {
  capturedHandler = null;
  NotificationCtor.mockClear();
  markedRead = [];
  server.use(
    getPostV2MarkSessionReadMockHandler(({ params }) => {
      markedRead.push(String(params.sessionId));
    }),
  );

  (
    FakeNotification as unknown as { permission: NotificationPermission }
  ).permission = "granted";
  vi.stubGlobal("Notification", FakeNotification);

  useCopilotUIStore.setState({
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

const completion = (sessionID: string) => ({
  type: "copilot_completion",
  event: "session_completed",
  session_id: sessionID,
});

describe("useCopilotNotifications — OS notification dispatch", () => {
  it("fires a browser notification with the AutoGPT/Task completed copy and 192px icon when a non-active session completes off-screen", async () => {
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });
    expect(capturedHandler).not.toBeNull();

    act(() => capturedHandler!(completion("sess-1")));

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

  it("notifies once for the same completion delivered twice", async () => {
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });

    act(() => capturedHandler!(completion("sess-1")));
    act(() => capturedHandler!(completion("sess-1")));
    act(() => capturedHandler!(completion("sess-2")));

    await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(2));
  });

  it("stays quiet for a replay of a turn that ended while the user watched", async () => {
    setVisibility("visible");
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });

    act(() => capturedHandler!({ ...completion("sess-1"), turn_id: "t1" }));
    setVisibility("hidden");
    act(() => capturedHandler!({ ...completion("sess-1"), turn_id: "t1" }));
    await act(async () => {});

    expect(NotificationCtor).not.toHaveBeenCalled();
  });

  it("tells a chat's next turn from a replay by its turn id", async () => {
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });

    act(() => capturedHandler!({ ...completion("sess-1"), turn_id: "t1" }));
    act(() => capturedHandler!({ ...completion("sess-1"), turn_id: "t1" }));
    act(() => capturedHandler!({ ...completion("sess-1"), turn_id: "t2" }));

    await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(2));
  });

  it("notifies again for a chat's next finished turn", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    try {
      renderHook(() => useCopilotNotifications(null), {
        wrapper: makeWrapper(),
      });

      act(() => capturedHandler!(completion("sess-1")));
      await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(1));

      vi.setSystemTime(Date.now() + 60_000);
      act(() => capturedHandler!(completion("sess-1")));
      await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(2));
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("useCopilotNotifications — marking the open chat read", () => {
  it("marks the chat read on the server when the user opens it", async () => {
    setVisibility("visible");
    setFocused(true);

    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(markedRead).toEqual(["sess-1"]));
  });

  it("marks it read again when a turn finishes while the user watches, without a notification", async () => {
    setVisibility("visible");
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(markedRead).toEqual(["sess-1"]));

    act(() => capturedHandler!(completion("sess-1")));

    await waitFor(() => expect(markedRead).toEqual(["sess-1", "sess-1"]));
    expect(NotificationCtor).not.toHaveBeenCalled();
  });

  it("leaves other chats unread", async () => {
    setVisibility("visible");
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(markedRead).toEqual(["sess-1"]));

    act(() => capturedHandler!(completion("sess-2")));
    await act(async () => {});

    expect(markedRead).toEqual(["sess-1"]);
  });
});

describe("useCopilotNotifications — a completion this tab never heard", () => {
  it("marks the open chat read when a list shows it unread", async () => {
    setVisibility("visible");
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await waitForFirstRead();

    // A turn ended while the socket was down: only the list's next read
    // shows it.
    act(() => {
      lastQueryClient.setQueryData([...SESSION_GROUPS_QUERY_KEY, {}], {
        status: 200,
        data: {
          groups: [
            {
              expert_id: null,
              total: 1,
              sessions: [
                { id: "sess-1", attention: "unread", is_unread: true },
              ],
            },
          ],
          pinned: [],
        },
      });
    });

    await waitFor(() => expect(markedRead).toEqual(["sess-1", "sess-1"]));
  });
});

describe("useCopilotNotifications — which cached list to believe", () => {
  function groups(session: Record<string, unknown>) {
    return {
      status: 200,
      data: {
        groups: [{ expert_id: null, total: 1, sessions: [session] }],
        pinned: [],
      },
    };
  }

  it("marks read an unseen turn under a needs-you marker", async () => {
    setVisibility("visible");
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await waitForFirstRead();

    act(() => {
      lastQueryClient.setQueryData(
        [...SESSION_GROUPS_QUERY_KEY, {}],
        groups({ id: "sess-1", attention: "needs_you", is_unread: true }),
      );
    });

    await waitFor(() => expect(markedRead).toEqual(["sess-1", "sess-1"]));
  });

  it("goes by the newest list when two disagree", async () => {
    setVisibility("visible");
    setFocused(true);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await waitForFirstRead();

    act(() => {
      lastQueryClient.setQueryData(
        [...SESSION_GROUPS_QUERY_KEY, "older"],
        groups({ id: "sess-1", attention: null, is_unread: false }),
        { updatedAt: Date.now() - 60_000 },
      );
      lastQueryClient.setQueryData(
        [...SESSION_GROUPS_QUERY_KEY, "newer"],
        groups({ id: "sess-1", attention: "unread", is_unread: true }),
      );
    });

    await waitFor(() => expect(markedRead).toEqual(["sess-1", "sess-1"]));
  });
});

describe("useCopilotNotifications — a read that fails", () => {
  it("sends it again rather than dropping it", async () => {
    let attempts = 0;
    server.use(
      http.post("*/api/chat/sessions/:sessionId/read", () => {
        attempts += 1;
        return attempts < 2
          ? HttpResponse.json({ detail: "unavailable" }, { status: 503 })
          : new HttpResponse(null, { status: 204 });
      }),
    );
    setVisibility("visible");
    setFocused(true);

    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(attempts).toBe(2), { timeout: 4_000 });
  });
});

describe("useCopilotNotifications — background tabs", () => {
  it("waits until the tab is visible before marking the open chat read", async () => {
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await act(async () => {});
    expect(markedRead).toEqual([]);

    setVisibility("visible");
    setFocused(true);
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await waitFor(() => expect(markedRead).toEqual(["sess-1"]));
  });
});

describe("useCopilotNotifications — unfocused windows", () => {
  it("waits until the window has focus before marking the open chat read", async () => {
    setVisibility("visible");
    setFocused(false);

    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await act(async () => {});
    expect(markedRead).toEqual([]);

    setFocused(true);
    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    await waitFor(() => expect(markedRead).toEqual(["sess-1"]));
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

    act(() => capturedHandler!(completion("sess-1")));
    await act(async () => {});

    expect(NotificationCtor).not.toHaveBeenCalled();
  });
});

describe("useCopilotNotifications — returning into an embedded frame", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("notices the user is back even when no focus event reaches the page", async () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    setVisibility("visible");
    setFocused(false);
    renderHook(() => useCopilotNotifications("sess-1"), {
      wrapper: makeWrapper(),
    });
    await act(async () => {});
    expect(markedRead).toEqual([]);

    // Focus returns straight into an iframe: the document has focus again,
    // but neither "focus" nor "visibilitychange" fires here.
    setFocused(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    // waitFor polls on an interval of its own.
    vi.useRealTimers();

    await waitFor(() => expect(markedRead).toEqual(["sess-1"]));
  });
});

describe("useCopilotNotifications — browser storage", () => {
  it("keeps no list of finished chats in the browser", async () => {
    renderHook(() => useCopilotNotifications(null), { wrapper: makeWrapper() });

    act(() => capturedHandler!(completion("sess-3")));
    await waitFor(() => expect(NotificationCtor).toHaveBeenCalledTimes(1));

    expect(
      window.localStorage.getItem("copilot-completed-sessions"),
    ).toBeNull();
  });
});
