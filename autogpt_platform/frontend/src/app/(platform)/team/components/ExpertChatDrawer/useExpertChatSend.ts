import { waitUntilChatSettled } from "@/app/(platform)/copilot/copilotChatRegistry";
import {
  useCopilotStreamStore,
  type PendingUploadSend,
} from "@/app/(platform)/copilot/copilotStreamStore";
import {
  describePendingUpload,
  sendWithAttachments,
  type OutgoingMessage,
} from "@/app/(platform)/copilot/helpers/sendWithAttachments";
import type { SendInput } from "@/app/(platform)/copilot/stream/turnRuntime";
import { recoverFailedDeferredSend } from "@/app/(platform)/copilot/useSendMessage";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import { useRef, useState } from "react";

interface Args {
  sessionId: string | null;
  sendMessage: (message: SendInput) => Promise<void>;
}

/**
 * Sends the drawer's messages through the copilot page's upload-then-send
 * path, and keeps the user's message on screen from the moment they press
 * Send: `hold` shows it while there is no session yet or it waits behind the
 * kickoff, and the session's placeholder takes over once it is sent, for as
 * long as its files upload or it waits for the reply in flight to end.
 */
export function useExpertChatSend({ sessionId, sendMessage }: Args) {
  const [held, setHeld] = useState<{
    message: OutgoingMessage;
    pending: PendingUploadSend;
  } | null>(null);
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const placeholder = useCopilotStreamStore((s) =>
    sessionId ? (s.pendingUploadSends[sessionId] ?? null) : null,
  );

  function hold(message: OutgoingMessage) {
    setHeld(
      (current) =>
        current ?? { message, pending: describePendingUpload(message) },
    );
  }

  function release(message: OutgoingMessage) {
    setHeld((current) => (current?.message === message ? null : current));
  }

  /**
   * The thread was reset: drops the held message and cancels the one still
   * on its way out of the thread being left, so it can never go out later,
   * from a Retry or a finished upload, into whichever thread shows by then.
   */
  function reset() {
    setHeld(null);
    const leftSessionId = sessionIdRef.current;
    if (!leftSessionId) return;
    const store = useCopilotStreamStore.getState();
    const unsent = store.pendingUploadSends[leftSessionId];
    if (!unsent) return;
    store.clearPendingUploadSend(leftSessionId, unsent);
    returnToComposer(unsent.text);
  }

  /** Sends through the drawer's chat, but only while it still shows the
   *  session the message was sent from. */
  function sendInSession(targetSessionId: string) {
    return async (input: SendInput) => {
      if (sessionIdRef.current !== targetSessionId) {
        throw new Error("The chat moved to another thread");
      }
      await sendMessageRef.current(input);
    };
  }

  /** Resolves once the reply to the message has streamed. */
  function sendNow(targetSessionId: string, message: OutgoingMessage) {
    const sent = sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: sendInSession(targetSessionId),
    });
    // The send above put the message on screen in this same tick, as the
    // placeholder or as the user's bubble, so the held copy can go.
    release(message);
    return sent;
  }

  /**
   * The pending buffer keeps a follow-up's files without their names, so the
   * transcript could not show them: a follow-up with files uploads them now
   * and goes out as the next turn once the reply in flight has ended. The
   * wait is on the session's chat, so a file retried after the panel was
   * closed and reopened still sends.
   */
  function sendAfterReply(targetSessionId: string, message: OutgoingMessage) {
    return sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: sendInSession(targetSessionId),
      waitUntilReady: () => waitUntilChatSettled(targetSessionId),
    });
  }

  return {
    pendingSend: placeholder ?? held?.pending ?? null,
    hold,
    release,
    reset,
    sendNow,
    sendAfterReply,
  };
}

/** Every message held in the drawer carries files, which cannot be put back
 *  in the composer, so only its text comes back. */
function returnToComposer(text: string) {
  if (text) useCopilotUIStore.getState().setInitialPrompt(text);
  toast({
    title: "Message not sent",
    description: text
      ? "You started a new task before it went out. Its text is back in the composer; attach the files again."
      : "You started a new task before it went out, so its files weren't sent.",
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
