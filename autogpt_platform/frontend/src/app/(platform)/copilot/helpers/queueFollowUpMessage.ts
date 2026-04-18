import { environment } from "@/services/environment";

import { getCopilotAuthHeaders } from "../helpers";

export type QueueFollowUpResult =
  | {
      kind: "queued";
      buffer_length: number;
      max_buffer_length: number;
      turn_in_flight: boolean;
    }
  | {
      kind: "raced_started_turn";
      status: number;
    };

/**
 * POST a follow-up message to the unified ``/stream`` endpoint when the
 * client believes a turn is already in flight.
 *
 * The server decides:
 *  - turn in flight → push to pending buffer, return ``202 JSON``  → kind="queued"
 *  - session idle  → start a new turn,         return ``200 SSE``  → kind="raced_started_turn"
 *
 * The "raced" branch covers a real race condition: the client's
 * ``isInflightRef`` reads true for one render, but by the time the request
 * lands the previous turn already finished.  Throwing in that case would
 * surface a misleading error toast — instead we drain the SSE response so
 * the connection is closed cleanly and signal "raced" to the caller, who
 * can rely on ``useHydrateOnStreamEnd`` to surface the new turn's output.
 *
 * Throws only on transport errors or unexpected non-200/202 status codes.
 */
export async function queueFollowUpMessage(
  sessionId: string,
  message: string,
): Promise<QueueFollowUpResult> {
  const url = `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(await getCopilotAuthHeaders()),
    },
    body: JSON.stringify({
      message,
      is_user_message: true,
      context: null,
      file_ids: null,
    }),
  });

  if (res.status === 202) {
    const body = (await res.json()) as {
      buffer_length: number;
      max_buffer_length: number;
      turn_in_flight: boolean;
    };
    return { kind: "queued", ...body };
  }

  if (res.status === 200) {
    // Race: server treated this as a fresh turn (previous turn finished
    // between our isInflight read and the request landing). Drain the body
    // so the underlying connection isn't held open, then signal the caller
    // — useHydrateOnStreamEnd will pick the assistant rows up next poll.
    try {
      await res.body?.cancel();
    } catch {
      // ignore — connection cleanup is best-effort
    }
    return { kind: "raced_started_turn", status: 200 };
  }

  throw new Error(
    `Expected 202 (queued) or 200 (raced new turn) from /stream; got ${res.status}`,
  );
}
