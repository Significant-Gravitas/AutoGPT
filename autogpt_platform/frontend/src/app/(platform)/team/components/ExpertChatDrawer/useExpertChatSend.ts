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

interface Args {
  sessionId: string | null;
  status: ChatStatus;
  sendMessage: (message: SendInput) => Promise<void>;
}

/**
 * Sends the drawer's messages through the copilot page's upload-then-send
 * path, and keeps the user's message on screen from the moment they press
 * Send: `hold` shows it while there is no session yet or it waits behind the
 * kickoff, and the session's placeholder takes over once it is sent, for as
 * long as its files upload or it waits for the reply in flight to end.
 */
export function useExpertChatSend({ sessionId, status, sendMessage }: Args) {
  const [heldMessage, setHeldMessage] = useState<OutgoingMessage | null>(null);
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;
  const { waitUntilSettled } = useSettleGate(
    status === "ready" || status === "error",
  );
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

  /** Resolves once the reply to the message has streamed. */
  function sendNow(targetSessionId: string, message: OutgoingMessage) {
    const sent = sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: latestSendMessage,
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
  function sendAfterReply(
    targetSessionId: string,
    message: OutgoingMessage,
    isStillCurrent: () => boolean,
  ) {
    return sendWithAttachments({
      sessionId: targetSessionId,
      message,
      sendMessage: latestSendMessage,
      waitUntilReady: async () => {
        if (!(await waitUntilSettled())) return false;
        if (isStillCurrent()) return true;
        returnToComposer(message);
        return false;
      },
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

/** The user started a new task while the message waited on the reply. */
function returnToComposer(message: OutgoingMessage) {
  if (message.text) useCopilotUIStore.getState().setInitialPrompt(message.text);
  toast({
    title: "Follow-up not sent",
    description: message.text
      ? "You started a new task before it could go out. It's back in the composer."
      : "You started a new task before it could go out, so its files weren't sent.",
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
