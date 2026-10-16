import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  describePendingUpload,
  sendWithAttachments,
  type OutgoingMessage,
} from "@/app/(platform)/copilot/helpers/sendWithAttachments";
import type { SendInput } from "@/app/(platform)/copilot/stream/turnRuntime";
import { recoverFailedDeferredSend } from "@/app/(platform)/copilot/useSendMessage";
import { useSettleGate } from "@/app/(platform)/copilot/useSettleGate";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import type { ChatStatus } from "ai";
import { useRef, useState } from "react";
import { useNextCommit } from "./useNextCommit";

interface Args {
  sessionId: string | null;
  status: ChatStatus;
  isReconnecting: boolean;
  isFinishProbing: boolean;
  sendMessage: (message: SendInput) => Promise<void>;
}

/**
 * Sends the drawer's messages through the copilot page's upload-then-send
 * path, and keeps the user's message on screen from the moment they press
 * Send: `hold` shows it while there is no session yet or it waits behind the
 * kickoff, and the session's placeholder takes over once it is sent, for as
 * long as its files upload or it waits for the reply in flight to end.
 */
export function useExpertChatSend({
  sessionId,
  status,
  isReconnecting,
  isFinishProbing,
  sendMessage,
}: Args) {
  const [heldMessage, setHeldMessage] = useState<OutgoingMessage | null>(null);
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;
  // The stream refuses a send while it reconnects, and a send during the
  // post-finish probe can still land in the backend's pending buffer.
  const isSettled =
    (status === "ready" || status === "error") &&
    !isReconnecting &&
    !isFinishProbing;
  const isSettledRef = useRef(isSettled);
  isSettledRef.current = isSettled;
  const { waitUntilSettled } = useSettleGate(isSettled);
  const nextCommit = useNextCommit();
  const placeholder = useCopilotStreamStore((s) =>
    sessionId ? (s.pendingUploadSends[sessionId] ?? null) : null,
  );

  function hold(message: OutgoingMessage) {
    setHeldMessage((current) => current ?? message);
  }

  /** Without a message, drops whatever is held (the thread was reset). */
  function release(message?: OutgoingMessage) {
    setHeldMessage((current) =>
      !message || current === message ? null : current,
    );
  }

  function latestSendMessage(input: SendInput) {
    return sendMessageRef.current(input);
  }

  /** Resolves `false` when the drawer closed first. */
  async function waitForReplyToEnd() {
    while (await waitUntilSettled()) {
      if (!(await nextCommit())) return false;
      if (isSettledRef.current) return true;
    }
    return false;
  }

  /** `latestSendMessage` sends into the thread on screen, so a message whose
   *  uploads outlived its own thread goes back to the composer instead. */
  function isStillShown(targetSessionId: string, message: OutgoingMessage) {
    if (sessionIdRef.current === targetSessionId) return true;
    returnToComposer(message);
    return false;
  }

  /** Resolves once the reply to the message has streamed. */
  function sendNow(targetSessionId: string, message: OutgoingMessage) {
    const sent = sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: latestSendMessage,
      waitUntilReady:
        message.files.length > 0
          ? async () => isStillShown(targetSessionId, message)
          : undefined,
    });
    // The send above put the message on screen in this same tick, as the
    // placeholder or as the user's bubble, so the held copy can go.
    release(message);
    return sent;
  }

  /**
   * The pending buffer keeps a follow-up's files without their names, so the
   * transcript could not show them: a follow-up with files uploads them now
   * and goes out as the next turn once the reply in flight has ended.
   */
  function sendAfterReply(targetSessionId: string, message: OutgoingMessage) {
    return sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: latestSendMessage,
      waitUntilReady: async () =>
        (await waitForReplyToEnd()) && isStillShown(targetSessionId, message),
    });
  }

  return {
    pendingSend:
      placeholder ?? (heldMessage ? describePendingUpload(heldMessage) : null),
    hold,
    release,
    sendNow,
    sendAfterReply,
  };
}

/** The user started a new task while the message's files uploaded or it
 *  waited on the reply. */
function returnToComposer(message: OutgoingMessage) {
  if (message.text) useCopilotUIStore.getState().setInitialPrompt(message.text);
  toast({
    title: "Message not sent",
    description: message.text
      ? "You started a new task before it went out. Its text is back in the composer; attach the files again."
      : "You started a new task before it went out. Attach the files again to send them.",
  });
}

/** For a send whose caller has already moved on. A kickoff prompt is ours,
 *  not the user's, so there is nothing to hand back for it. */
export function recoverUnsentMessage(message: OutgoingMessage, error: unknown) {
  if (message.metadata) {
    Sentry.captureException(error);
    return;
  }
  recoverFailedDeferredSend(message.text, [], error);
}
