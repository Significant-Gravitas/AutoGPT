import { server } from "@/mocks/mock-server";
import { renderHook, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { getStoppableSubSessionIds, useCopilotStop } from "../useCopilotStop";

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));

function delegate(id: string, output: unknown): UIMessage {
  return {
    id: `m-${id}`,
    role: "assistant",
    parts: [
      {
        type: "tool-delegate_to_expert",
        state: "output-available",
        toolCallId: id,
        input: { expert_id: "exp-alex" },
        output,
      },
    ],
  } as unknown as UIMessage;
}

const MESSAGES = [
  delegate("c1", { status: "running", sub_session_id: "sub-1" }),
  delegate("c2", { status: "queued", sub_session_id: "sub-2" }),
  delegate("c3", { status: "completed", sub_session_id: "sub-3" }),
  delegate("c4", { status: "running", sub_session_id: "sub-1" }),
];

function cancelRoute(failing: string[] = []) {
  const cancelled: string[] = [];
  server.use(
    http.post("*/api/chat/sessions/:sessionId/cancel", ({ params }) => {
      const id = String(params.sessionId);
      cancelled.push(id);
      if (failing.includes(id)) return new HttpResponse(null, { status: 500 });
      return HttpResponse.json({ cancelled: true, reason: "ok" });
    }),
  );
  return cancelled;
}

function renderStop(messages: UIMessage[]) {
  const { result } = renderHook(() =>
    useCopilotStop({
      sessionId: "chat-1",
      sdkStop: vi.fn(),
      setMessages: vi.fn(),
      isUserStoppingRef: { current: false },
      setIsUserStopping: vi.fn(),
      messages,
    }),
  );
  return result.current;
}

describe("useCopilotStop", () => {
  it("picks each in-flight hand-off's sub-session once", () => {
    expect(getStoppableSubSessionIds(MESSAGES)).toEqual(["sub-1", "sub-2"]);
  });

  it("stops the teammates along with the chat", async () => {
    const cancelled = cancelRoute();
    await renderStop(MESSAGES)();
    await waitFor(() =>
      expect([...cancelled].sort()).toEqual(["chat-1", "sub-1", "sub-2"]),
    );
    expect(toast).not.toHaveBeenCalled();
  });

  it("raises one toast when a teammate could not be stopped", async () => {
    toast.mockReset();
    cancelRoute(["sub-1", "sub-2"]);
    await renderStop(MESSAGES)();
    await waitFor(() => expect(toast).toHaveBeenCalledTimes(1));
    expect(toast.mock.calls[0][0].title).toBe("Could not stop 2 experts");
  });
});
