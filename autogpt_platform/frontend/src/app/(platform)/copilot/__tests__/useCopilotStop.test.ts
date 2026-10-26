import { server } from "@/mocks/mock-server";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStop } from "../useCopilotStop";

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));

/** Sessions by id: which are live; every cancel is recorded. */
function sessions(live: string[], failing: string[] = []) {
  const cancelled: string[] = [];
  server.use(
    http.get("*/api/chat/sessions/:sessionId", ({ params }) =>
      HttpResponse.json({
        id: String(params.sessionId),
        created_at: "2026-09-28T00:00:00Z",
        updated_at: "2026-09-28T00:00:00Z",
        user_id: "u-1",
        chat_status: live.includes(String(params.sessionId))
          ? "running"
          : "idle",
        messages: [],
      }),
    ),
    http.post("*/api/chat/sessions/:sessionId/cancel", ({ params }) => {
      const id = String(params.sessionId);
      cancelled.push(id);
      if (failing.includes(id)) return new HttpResponse(null, { status: 500 });
      return HttpResponse.json({ cancelled: true, reason: "ok" });
    }),
  );
  return cancelled;
}

function renderStop() {
  const { result } = renderHook(() =>
    useCopilotStop({
      sessionId: "chat-1",
      stopStream: vi.fn(),
      isUserStoppingRef: { current: false },
      setIsUserStopping: vi.fn(),
    }),
  );
  return result.current;
}

describe("stopping the chat", () => {
  beforeEach(() => toast.mockReset());

  it("stops only this chat's turn; the server stops the experts it set working", async () => {
    // The experts' threads are live, but a turn running there may be one the
    // user started in that expert's own thread after the hand-off finished.
    // Only the server can tell, from who sent the message that turn runs on,
    // so the browser cancels nothing but its own chat.
    const cancelled = sessions(["sub-1", "sub-2", "sub-old"]);
    await renderStop()();
    await waitFor(() => expect(cancelled).toContain("chat-1"));
    expect(cancelled).toEqual(["chat-1"]);
    expect(toast).not.toHaveBeenCalled();
  });
});
