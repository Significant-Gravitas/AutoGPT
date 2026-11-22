import {
  forgetHeldFollowUp,
  rememberHeldFollowUp,
  takeHeldFollowUps,
} from "@/app/(platform)/copilot/helpers/heldFollowUps";
import { toastSendFailure } from "@/app/(platform)/copilot/helpers/sendFailureToast";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useRef, useState } from "react";

export interface HeldFollowUp {
  id: number;
  sessionId: string;
  /** Where its copy waits in sessionStorage. */
  heldKey: string;
  text: string;
}

interface Entry extends HeldFollowUp {
  isQueuing: boolean;
}

/**
 * Follow-ups the builder chat holds until its stream has settled (see
 * `useBuilderChatSend`), shown as queued chips. Each also waits in
 * sessionStorage, because the panel unmounts when the user leaves the
 * builder and is gone on a reload: one left behind that way goes back to the
 * composer the next time its chat opens.
 *
 * A copy waits under its session unless `heldKey` names another key: the
 * memory chat starts a new session on every open, so its copies wait for the
 * next chat of the same memory instead.
 */
export function useHeldFollowUps(
  sessionId: string | null,
  heldKey: string | null = sessionId,
) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const nextIdRef = useRef(0);

  useEffect(() => {
    if (!sessionId || !heldKey) return;
    const leftBehind = takeHeldFollowUps(heldKey);
    if (leftBehind.length === 0) return;
    returnToComposer(
      leftBehind.join("\n\n"),
      "You left the chat before it could go out.",
    );
  }, [sessionId, heldKey]);

  function hold(forSessionId: string, text: string) {
    const followUp: HeldFollowUp = {
      id: nextIdRef.current++,
      sessionId: forSessionId,
      heldKey: heldKey ?? forSessionId,
      text,
    };
    setEntries((prev) => [...prev, { ...followUp, isQueuing: false }]);
    rememberHeldFollowUp(followUp.heldKey, text);
    return followUp;
  }

  /** While its queue request is out, its own chip stands in for it. */
  function setQueuing(followUp: HeldFollowUp, isQueuing: boolean) {
    setEntries((prev) =>
      prev.map((entry) =>
        entry.id === followUp.id ? { ...entry, isQueuing } : entry,
      ),
    );
  }

  /** It went out, was queued, or went back to the composer. */
  function release(followUp: HeldFollowUp) {
    drop(followUp);
    forgetHeldFollowUp(followUp.heldKey, followUp.text);
  }

  /**
   * Another workflow's chat is on screen, so the composer is not its own: it
   * keeps its stored copy, which goes back when a chat under its `heldKey`
   * opens again.
   */
  function setAside(followUp: HeldFollowUp) {
    drop(followUp);
  }

  function drop(followUp: HeldFollowUp) {
    setEntries((prev) => prev.filter((entry) => entry.id !== followUp.id));
  }

  return {
    texts: entries
      .filter((entry) => entry.sessionId === sessionId && !entry.isQueuing)
      .map((entry) => entry.text),
    hold,
    setQueuing,
    release,
    setAside,
  };
}

/** For a follow-up that never went out: its text goes back in the composer. */
export function returnToComposer(text: string, reason: string) {
  useCopilotUIStore.getState().restoreDraft(text);
  toast({
    title: "Follow-up not sent",
    description: `${reason} It's back in the composer.`,
  });
}

/** For a held follow-up that failed once the composer had moved on. */
export function returnFailedFollowUp(text: string, error: unknown) {
  useCopilotUIStore.getState().restoreDraft(text);
  toastSendFailure({
    title: "Couldn't send message",
    error,
    recovery: "your message is back in the composer",
  });
}

/** For a follow-up set aside for its own chat, which is no longer on screen. */
export function noteSetAside(reason: string) {
  toast({
    title: "Follow-up not sent",
    description: `${reason} It will be back in its chat's composer.`,
  });
}

/** For a set-aside follow-up whose queue request failed. */
export function noteFailedSetAside(error: unknown) {
  toastSendFailure({
    title: "Couldn't send message",
    error,
    recovery: "your message will be back in its chat's composer",
  });
}
