import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type WSHandler = (notification: unknown) => void;
let capturedHandler: WSHandler | null = null;
let mockPathname = "/library";
let mockSearch = "";

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketMessage: (_event: string, handler: WSHandler) => {
      capturedHandler = handler;
      return () => {};
    },
  }),
  BackendAPIProvider: ({ children }: { children: ReactNode }) => children,
}));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  usePathname: () => mockPathname,
  useSearchParams: () => new URLSearchParams(mockSearch),
}));

import { useCopilotUIStore } from "../../../store";
import { CopilotNotificationsListener } from "../CopilotNotificationsListener";

function renderListener() {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <CopilotNotificationsListener />
    </QueryClientProvider>,
  );
}

function finish(sessionID: string) {
  act(() => {
    capturedHandler!({
      type: "copilot_completion",
      event: "session_completed",
      session_id: sessionID,
    });
  });
}

beforeEach(() => {
  capturedHandler = null;
  useCopilotUIStore.setState({
    completedSessionIDs: new Set(),
    isNotificationsEnabled: false,
  });
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => "visible",
  });
  vi.spyOn(document, "hasFocus").mockReturnValue(true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  mockPathname = "/library";
  mockSearch = "";
});

describe("CopilotNotificationsListener", () => {
  it("marks a chat finished while the user is on another page", async () => {
    renderListener();

    finish("sess-1");

    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs).toContain(
        "sess-1",
      );
    });
  });

  it("treats a chat open on /home as seen", async () => {
    mockPathname = "/home";
    mockSearch = "sessionId=sess-1";
    renderListener();

    finish("sess-1");
    finish("sess-2");

    await waitFor(() => {
      expect(useCopilotUIStore.getState().completedSessionIDs).toEqual(
        new Set(["sess-2"]),
      );
    });
  });
});
