import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import { useCopilotPendingChips } from "@/app/(platform)/copilot/useCopilotPendingChips";
import { useSettleGate } from "@/app/(platform)/copilot/useSettleGate";
import * as Sentry from "@sentry/nextjs";
import type { ChatStatus } from "ai";
import { useInsertionEffect, useRef } from "react";
import { isQueueRefusal } from "./helpers";
import {
  returnFailedFollowUp,
  returnToComposer,
  useHeldFollowUps,
  type HeldFollowUp,
} from "./useHeldFollowUps";

type ChipArgs = Parameters<typeof useCopilotPendingChips>[0];

interface Args {
  sessionId: string | null;
  status: ChatStatus;
  isReconnecting: boolean;
  isFinishProbing: boolean;
  messages: ChipArgs["messages"];
  setMessages: NonNullable<ChipArgs["setMessages"]>;
  sendMessage: (message: { text: string }) => unknown;
}

/**
 * Sends a message, or queues it behind the turn in flight. A follow-up the
 * backend refuses to queue (its turn has already ended) is held, shown as a
 * queued chip, and goes out as a new turn once this panel has finished
 * drawing the previous answer.
 */
export function useBuilderChatSend({
  sessionId,
  status,
  isReconnecting,
  isFinishProbing,
  messages,
  setMessages,
  sendMessage,
}: Args) {
  // Refs, not props: a follow-up that waited for the settle must see a turn
  // another held follow-up dispatched in the same settle, and the workflow
  // the panel shows now rather than the one it was typed on.
  const isInFlight = status === "streaming" || status === "submitted";
  const isInFlightRef = useRef(isInFlight);
  const sessionIdRef = useRef(sessionId);
  useInsertionEffect(() => {
    isInFlightRef.current = isInFlight;
    sessionIdRef.current = sessionId;
  });
  // The server's turn can end before the screen does: the smoothing
  // transform paces text out word by word, and the post-finish probe may
  // still turn into a reconnect. Sending while this stream is live cuts the
  // answer off mid-sentence. Scoped to the session, because opening another
  // workflow swaps it without remounting this hook.
  const { waitUntilSettled, markUnsettled } = useSettleGate(
    !isInFlight && !isFinishProbing && !isReconnecting,
    sessionId,
  );
  const chips = useCopilotPendingChips({
    sessionId,
    status,
    messages,
    setMessages,
  });
  const held = useHeldFollowUps(sessionId);

  async function onSend(message: string, _files?: File[]) {
    const text = message.trim();
    if (!text || !sessionId) return;
    if (!isInFlightRef.current) {
      dispatch(text);
      return;
    }
    try {
      if (await queueBehindTurn(sessionId, text)) return;
    } catch (error) {
      Sentry.captureException(error);
      // The composer restores the draft and shows the one toast for it.
      throw error;
    }
    // Not awaited: the composer keeps Enter and the queue button locked
    // until this resolves, and the wait can last as long as the answer.
    void sendOnceSettled(held.hold(sessionId, text));
  }

  /**
   * Resolves false, with its chip taken down again, when the backend refused
   * it because its turn had already ended. The chip goes up before the
   * request, so a peek that answers first counts it rather than restoring
   * the message under a second chip.
   */
  async function queueBehindTurn(forSessionId: string, text: string) {
    const chipId = chips.queueMessage(text);
    try {
      await queueFollowUpMessage(forSessionId, text);
      return true;
    } catch (error) {
      chips.unqueueMessage(chipId);
      if (isQueueRefusal(error)) return false;
      throw error;
    }
  }

  /**
   * Sends a held follow-up as a new turn once this panel's stream settles.
   * If the panel unmounts first, its stored copy goes back to the composer
   * the next time this chat opens; if the panel moves to another workflow,
   * it goes back now.
   */
  async function sendOnceSettled(followUp: HeldFollowUp) {
    for (;;) {
      const isSettled =
        sessionIdRef.current === followUp.sessionId &&
        (await waitUntilSettled());
      if (sessionIdRef.current !== followUp.sessionId) {
        held.release(followUp);
        returnToComposer(
          followUp.text,
          "You opened another workflow before it went out.",
        );
        return;
      }
      if (!isSettled) return;
      if (!isInFlightRef.current) {
        held.release(followUp);
        dispatch(followUp.text);
        return;
      }
      // Another held follow-up started a turn in this same settle.
      if (await queueHeldBehindTurn(followUp)) return;
    }
  }

  /** Resolves false when the backend refused it again. */
  async function queueHeldBehindTurn(followUp: HeldFollowUp) {
    held.setQueuing(followUp, true);
    try {
      if (!(await queueBehindTurn(followUp.sessionId, followUp.text))) {
        held.setQueuing(followUp, false);
        return false;
      }
      held.release(followUp);
    } catch (error) {
      Sentry.captureException(error);
      held.release(followUp);
      returnFailedFollowUp(followUp.text, error);
    }
    return true;
  }

  function dispatch(text: string) {
    isInFlightRef.current = true;
    markUnsettled();
    sendMessage({ text });
  }

  return {
    onSend,
    queuedMessages:
      held.texts.length > 0
        ? [...chips.queuedMessages, ...held.texts]
        : chips.queuedMessages,
  };
}
