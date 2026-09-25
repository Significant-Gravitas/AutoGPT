import { environment } from "@/services/environment";
import { v4 as uuidv4 } from "uuid";

import { getCopilotAuthHeaders } from "../helpers";

export interface QueueFollowUpResult {
  buffer_length: number;
  max_buffer_length: number;
  turn_in_flight: boolean;
}

export class QueueFollowUpNotActiveError extends Error {
  constructor() {
    super("Session has no active turn to queue against.");
    this.name = "QueueFollowUpNotActiveError";
  }
}

/**
 * POST a follow-up message to the dedicated pending-message endpoint when
 * the client believes a turn is already in flight.
 */
export async function queueFollowUpMessage(
  sessionId: string,
  message: string,
): Promise<QueueFollowUpResult> {
  const url = `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/messages/pending`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(await getCopilotAuthHeaders()),
    },
    body: JSON.stringify({
      message,
      context: null,
      file_ids: null,
      // Idempotency key: a retransmit of this request is queued only once.
      // Options force uuid's getRandomValues path, which (unlike
      // crypto.randomUUID) exists on plain-HTTP LAN origins.
      message_id: uuidv4({}),
    }),
  });

  if (res.status === 200) {
    const result = (await res.json()) as QueueFollowUpResult;
    if (!result.turn_in_flight) {
      throw new QueueFollowUpNotActiveError();
    }
    return result;
  }

  if (res.status === 409) {
    throw new QueueFollowUpNotActiveError();
  }

  throw new Error(`Expected 200 from /messages/pending; got ${res.status}`);
}
