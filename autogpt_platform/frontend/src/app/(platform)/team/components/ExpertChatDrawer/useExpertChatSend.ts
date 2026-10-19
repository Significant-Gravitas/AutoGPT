import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  describePendingUpload,
  hasAttachments,
  sendWithAttachments,
  type OutgoingMessage,
} from "@/app/(platform)/copilot/helpers/sendWithAttachments";
import type { SendInput } from "@/app/(platform)/copilot/stream/turnRuntime";
import { toastSendFailure } from "@/app/(platform)/copilot/helpers/sendFailureToast";
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
  stop: () => Promise<void>;
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
  stop,
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
  // Bumped by every Stop. The backend drops the follow-ups it buffered when
  // a reply is cancelled, so one held here for the reply drops with them.
  const stopCountRef = useRef(0);
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

  function stopReply() {
    stopCountRef.current += 1;
    return stop();
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
    returnToComposer(message, "new-task");
    return false;
  }

  function wasNotStopped(stopCount: number, message: OutgoingMessage) {
    if (stopCountRef.current === stopCount) return true;
    returnToComposer(message, "stopped");
    return false;
  }

  /**
   * Resolves once the reply to the message has streamed. A message with
   * files, or with workspace files, goes out only once the turn in flight has
   * ended: the pending buffer keeps a follow-up's files without their names,
   * so the transcript could not show them, and the stream refuses a send
   * while it reconnects without saying so to its caller.
   */
  function send(targetSessionId: string, message: OutgoingMessage) {
    const stopCount = stopCountRef.current;
    const sent = sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: latestSendMessage,
      waitUntilReady: hasAttachments(message)
        ? async () =>
            isStillShown(targetSessionId, message) &&
            (await waitForReplyToEnd()) &&
            isStillShown(targetSessionId, message) &&
            wasNotStopped(stopCount, message)
        : undefined,
    });
    // The send above put the message on screen in this same tick, as the
    // placeholder or as the user's bubble, so the held copy can go.
    release(message);
    return sent;
  }

  return {
    pendingSend:
      placeholder ?? (heldMessage ? describePendingUpload(heldMessage) : null),
    hold,
    release,
    send,
    stop: stopReply,
  };
}

const UNSENT_BECAUSE = {
  "new-task": "You started a new task before it went out.",
  stopped: "You stopped the reply before it went out.",
};

/** The user started a new task, or stopped the reply, while the message's
 *  files uploaded or it waited on the reply. */
function returnToComposer(
  message: OutgoingMessage,
  reason: keyof typeof UNSENT_BECAUSE,
) {
  if (message.text) useCopilotUIStore.getState().restoreDraft(message.text);
  toast({
    title: "Message not sent",
    description: message.text
      ? `${UNSENT_BECAUSE[reason]} Its text is back in the composer; attach the files again.`
      : `${UNSENT_BECAUSE[reason]} Attach the files again to send them.`,
  });
}

/** For a send whose caller has already moved on. A kickoff prompt is ours,
 *  not the user's, so there is nothing to hand back for it. */
export function recoverUnsentMessage(message: OutgoingMessage, error: unknown) {
  if (message.metadata) {
    Sentry.captureException(error);
    return;
  }
  if (message.text) useCopilotUIStore.getState().restoreDraft(message.text);
  toastSendFailure({
    title: "Couldn't send message",
    error,
    recovery: message.text
      ? "your message is back in the composer"
      : "your files were not sent",
  });
}
