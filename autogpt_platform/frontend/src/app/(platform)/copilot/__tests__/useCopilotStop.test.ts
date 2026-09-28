import { server } from "@/mocks/mock-server";
import { renderHook, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getStoppableSubSessionIds } from "../stopCascade";
import { useCopilotStop } from "../useCopilotStop";

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

function user(id: string, metadata?: unknown): UIMessage {
  return {
    id,
    role: "user",
    metadata,
    parts: [{ type: "text", text: "hi" }],
  } as UIMessage;
}

const OLD_TURN = [
  user("u-old"),
  delegate("c-old", { status: "running", sub_session_id: "sub-old" }),
];

const MESSAGES = [
  ...OLD_TURN,
  user("u-now"),
  delegate("c1", { status: "running", sub_session_id: "sub-1" }),
  delegate("c2", { status: "queued", sub_session_id: "sub-2" }),
  delegate("c3", { status: "completed", sub_session_id: "sub-3" }),
  delegate("c4", { status: "running", sub_session_id: "sub-1" }),
];

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

describe("stopping the teammates with the chat", () => {
  beforeEach(() => toast.mockReset());

  it("picks this turn's in-flight hand-offs once each", () => {
    expect(getStoppableSubSessionIds(MESSAGES)).toEqual(["sub-1", "sub-2"]);
  });

  it("keeps a hand-off approved mid-turn in the turn", () => {
    const messages = [
      user("u-now"),
      delegate("c1", { type: "approval_required", review_id: "rev-1" }),
      user("held", {
        held_call: {
          tool_call_id: "c1",
          review_id: "rev-1",
          outcome: "approved",
        },
      }),
    ];
    messages[2] = {
      ...messages[2],
      parts: [
        {
          type: "text",
          text: `<held_call_result>\n${JSON.stringify({ status: "running", sub_session_id: "sub-9" })}\n</held_call_result>`,
        },
      ],
    } as UIMessage;
    expect(getStoppableSubSessionIds(messages)).toEqual(["sub-9"]);
  });

  it("stops a live teammate from this turn and leaves an old turn's alone", async () => {
    const cancelled = sessions(["sub-1", "sub-old"]);
    await renderStop(MESSAGES)();
    await waitFor(() => expect(cancelled).toContain("sub-1"));
    expect(cancelled).not.toContain("sub-old");
    // Transcript said queued, but the teammate's session is idle.
    expect(cancelled).not.toContain("sub-2");
    expect(toast).not.toHaveBeenCalled();
  });

  it("raises one toast when a teammate could not be stopped", async () => {
    sessions(["sub-1", "sub-2"], ["sub-1", "sub-2"]);
    await renderStop(MESSAGES)();
    await waitFor(() => expect(toast).toHaveBeenCalledTimes(1));
    expect(toast.mock.calls[0][0].title).toBe("Could not stop 2 experts");
  });
});
