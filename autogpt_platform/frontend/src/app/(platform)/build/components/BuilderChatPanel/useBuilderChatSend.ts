import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import { useSettleGate } from "@/app/(platform)/copilot/useSettleGate";
import * as Sentry from "@sentry/nextjs";
import type { ChatStatus } from "ai";
import { useInsertionEffect, useRef } from "react";

interface Args {
  sessionId: string | null;
  status: ChatStatus;
  isReconnecting: boolean;
  isFinishProbing: boolean;
  sendMessage: (message: { text: string }) => unknown;
  queueMessage: (text: string) => void;
}

/**
 * Sends a message, or queues it behind the turn in flight. A follow-up the
 * backend refuses to queue (its turn has already ended) goes out as a new
 * turn once this panel has finished drawing the previous answer.
 */
export function useBuilderChatSend({
  sessionId,
  status,
  isReconnecting,
  isFinishProbing,
  sendMessage,
  queueMessage,
}: Args) {
  // A ref, not `status`: a follow-up that waited for the settle must see a
  // turn another held follow-up dispatched in the same settle.
  const isInFlight = status === "streaming" || status === "submitted";
  const isInFlightRef = useRef(isInFlight);
  useInsertionEffect(() => {
    isInFlightRef.current = isInFlight;
  });
  // The server's turn can end before the screen does: the smoothing
  // transform paces text out word by word, and the post-finish probe may
  // still turn into a reconnect. Sending while this stream is live cuts the
  // answer off mid-sentence.
  const { waitUntilSettled, markUnsettled } = useSettleGate(
    !isInFlight && !isFinishProbing && !isReconnecting,
  );

  async function onSend(message: string, _files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed) return;
    if (!sessionId) return;
    while (isInFlightRef.current) {
      try {
        await queueFollowUpMessage(sessionId, trimmed);
        queueMessage(trimmed);
        return;
      } catch (err) {
        if (
          !(err instanceof Error && err.name === "QueueFollowUpNotActiveError")
        ) {
          Sentry.captureException(err);
          // The composer restores the draft and shows the one toast for it.
          throw err;
        }
      }
      if (!(await waitUntilSettled())) return;
    }
    isInFlightRef.current = true;
    markUnsettled();
    sendMessage({ text: trimmed });
  }

  return { onSend };
}
