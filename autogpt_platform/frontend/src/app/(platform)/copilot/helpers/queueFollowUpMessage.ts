import { environment } from "@/services/environment";

import { getCopilotAuthHeaders } from "../helpers";

/**
 * POST a follow-up message to the unified ``/stream`` endpoint when the
 * session already has a turn in flight.
 *
 * The server detects the in-flight turn, pushes the message into the pending
 * buffer, and returns ``202 application/json`` with the buffer state.  We
 * can't use the AI SDK's ``sendMessage`` for this case because it's wired
 * to expect ``text/event-stream`` — hitting it with a 202 JSON response
 * would make the SDK's stream parser throw.
 *
 * Throws on network error or non-202 status.
 */
export async function queueFollowUpMessage(
  sessionId: string,
  message: string,
): Promise<{
  buffer_length: number;
  max_buffer_length: number;
  turn_in_flight: boolean;
}> {
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

  if (res.status !== 202) {
    throw new Error(
      `Expected 202 queued response from /stream; got ${res.status}`,
    );
  }
  return res.json();
}
