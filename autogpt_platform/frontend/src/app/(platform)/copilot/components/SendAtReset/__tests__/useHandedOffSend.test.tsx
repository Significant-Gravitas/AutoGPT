import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { StrictMode, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatDraft } from "../../ChatInput/useChatDraft";
import { sessionDraftKey } from "../helpers";
import { useScheduledSendHandoff } from "../scheduledSendHandoff";
import { useHandedOffSend } from "../useHandedOffSend";
import { useScheduledSend } from "../useScheduledSend";

vi.mock("@/lib/auth/hooks/useAuthStore", () => ({
  useAuthStore: (selector: (state: { user: { id: string } }) => unknown) =>
    selector({ user: { id: "user-a" } }),
}));

const RUN_AT = new Date(2026, 9, 17, 1, 7).toISOString();
const MESSAGE = "Draft the launch post";

function withQueryClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <StrictMode>
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      </StrictMode>
    );
  };
}

function useNewChat(sessionId: string) {
  const scheduled = useScheduledSend({ sessionId });
  useHandedOffSend(sessionId, scheduled);
  const [draft] = useChatDraft(sessionDraftKey(sessionId));
  return draft;
}

function serverHolding(waiting: unknown[] | "unavailable") {
  const calls = { posted: [] as unknown[] };
  server.use(
    http.get("*/api/chat/sessions/:sessionId/at-reset", () =>
      waiting === "unavailable"
        ? HttpResponse.json(
            { detail: "Scheduler unavailable" },
            { status: 503 },
          )
        : HttpResponse.json({ jobs: waiting }),
    ),
    http.post(
      "*/api/chat/sessions/:sessionId/at-reset",
      async ({ request }) => {
        const body = (await request.json()) as { message: string };
        calls.posted.push(body);
        return HttpResponse.json({
          kind: "send",
          run_at: RUN_AT,
          message: body.message,
        });
      },
    ),
  );
  return calls;
}

function handedOff(sessionId: string) {
  return useScheduledSendHandoff.getState().sends[sessionId];
}

beforeEach(() => {
  sessionStorage.clear();
  useScheduledSendHandoff.setState({ sends: {} });
});

describe("useHandedOffSend", () => {
  it("sets a send the new-chat screen handed over once, then lets it go", async () => {
    const calls = serverHolding([]);
    useScheduledSendHandoff.getState().hold("chat-new", MESSAGE);

    renderHook(() => useNewChat("chat-new"), { wrapper: withQueryClient() });

    await waitFor(() => expect(handedOff("chat-new")).toBeUndefined());
    expect(calls.posted).toEqual([{ kind: "send", message: MESSAGE }]);
  });

  describe("after a reload while the send was being set", () => {
    function sentBeforeTheReload(sessionId: string) {
      useScheduledSendHandoff.setState({
        sends: { [sessionId]: { message: MESSAGE, sentAt: Date.now() } },
      });
    }

    it("keeps it when the server has it, joined onto what was waiting", async () => {
      const calls = serverHolding([
        { kind: "send", run_at: RUN_AT, message: `Earlier note\n\n${MESSAGE}` },
      ]);
      sentBeforeTheReload("chat-landed");

      const { result } = renderHook(() => useNewChat("chat-landed"), {
        wrapper: withQueryClient(),
      });

      await waitFor(() => expect(handedOff("chat-landed")).toBeUndefined());
      expect(calls.posted).toEqual([]);
      expect(result.current).toBe("");
    });

    it("puts it back in the composer when the server doesn't have it", async () => {
      const calls = serverHolding([]);
      sentBeforeTheReload("chat-lost");

      const { result } = renderHook(() => useNewChat("chat-lost"), {
        wrapper: withQueryClient(),
      });

      await waitFor(() => expect(result.current).toBe(MESSAGE));
      expect(handedOff("chat-lost")).toBeUndefined();
      expect(calls.posted).toEqual([]);
    });

    it("puts it back in the composer when the server can't be asked", async () => {
      const calls = serverHolding("unavailable");
      sentBeforeTheReload("chat-unsure");

      const { result } = renderHook(() => useNewChat("chat-unsure"), {
        wrapper: withQueryClient(),
      });

      await waitFor(() => expect(result.current).toBe(MESSAGE));
      expect(calls.posted).toEqual([]);
    });
  });
});
