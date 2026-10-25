import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { http, HttpResponse } from "msw";

export const HELD_FOLLOW_UP = "What is my team working on right now?";
export const RETRY_ANSWER = "Maya is drafting the launch plan.";
const HELD_ID = "pm-1";

interface StreamBody {
  message?: string;
  message_id?: string;
  resend_pending_id?: string | null;
}

interface Args {
  baseUrl: string;
  sessionId: string;
  /** "load": held already, as after a reload. "first turn": the chat's first
   *  turn ends before reading a follow-up typed during it. */
  heldFrom: "load" | "first turn";
  session?: Partial<SessionDetailResponse>;
}

/**
 * The server side of a chat with a follow-up its last turn ended without
 * reading. The pending queue reports it held, by its id, until a Retry names
 * that id; the resend then runs as a turn of its own and the queue is empty.
 */
export function heldFollowUpServer({
  baseUrl,
  sessionId,
  heldFrom,
  session,
}: Args) {
  const state = {
    held: heldFrom === "load",
    bodies: [] as StreamBody[],
    stored: [] as SessionDetailResponseMessagesItem[],
  };

  function store(role: "user" | "assistant", content: string) {
    const sequence = state.stored.length + 1;
    state.stored = [
      ...state.stored,
      {
        id: `db-${sequence}`,
        role,
        content,
        tool_call_id: null,
        tool_calls: null,
        sequence,
        duration_ms: null,
        created_at: `2026-10-25T00:${String(sequence).padStart(2, "0")}:00Z`,
        metadata: null,
      },
    ];
  }

  const handlers = [
    http.get(`*/api/chat/sessions/${sessionId}`, () =>
      HttpResponse.json({
        id: sessionId,
        created_at: "2026-10-25T00:00:00Z",
        updated_at: "2026-10-25T00:00:00Z",
        user_id: "test-user",
        chat_status: "idle",
        messages: state.stored,
        has_more_messages: false,
        oldest_sequence: null,
        active_stream: null,
        metadata: { dry_run: false, builder_graph_id: null },
        expert_id: null,
        ...session,
      }),
    ),
    http.get(`*/api/chat/sessions/${sessionId}/messages/pending`, () =>
      HttpResponse.json(
        state.held
          ? {
              messages: [HELD_FOLLOW_UP],
              count: 1,
              items: [{ id: HELD_ID, content: HELD_FOLLOW_UP }],
              held_reason: "error",
            }
          : { messages: [], count: 0, items: [], held_reason: null },
      ),
    ),
    http.post(
      `${baseUrl}/api/chat/sessions/${sessionId}/stream`,
      async ({ request }) => {
        const body = (await request.json()) as StreamBody;
        state.bodies.push(body);
        const isResend = body.resend_pending_id === HELD_ID;
        const reply = isResend ? RETRY_ANSWER : "Here is what I found.";
        store("user", isResend ? HELD_FOLLOW_UP : (body.message ?? ""));
        store("assistant", reply);
        // A resend takes the follow-up off the queue; any other turn ends
        // before reading it.
        state.held = !isResend;
        const turn = state.bodies.length;
        return streamSseResponse(
          assistantTextChunks(reply, {
            messageId: `answer-${turn}`,
            textPartId: `answer-${turn}-text`,
          }),
          { abortSignal: request.signal },
        );
      },
    ),
  ];

  return {
    handlers,
    resends: () => state.bodies.filter((body) => body.resend_pending_id),
  };
}
