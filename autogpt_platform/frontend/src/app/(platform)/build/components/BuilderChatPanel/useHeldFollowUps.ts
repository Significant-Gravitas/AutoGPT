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
 */
export function useHeldFollowUps(sessionId: string | null) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const nextIdRef = useRef(0);

  useEffect(() => {
    if (!sessionId) return;
    const leftBehind = takeHeldFollowUps(sessionId);
    if (leftBehind.length === 0) return;
    returnToComposer(
      leftBehind.join("\n\n"),
      "You left the chat before it could go out.",
    );
  }, [sessionId]);

  function hold(forSessionId: string, text: string) {
    const followUp: HeldFollowUp = {
      id: nextIdRef.current++,
      sessionId: forSessionId,
      text,
    };
    setEntries((prev) => [...prev, { ...followUp, isQueuing: false }]);
    rememberHeldFollowUp(forSessionId, text);
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
    setEntries((prev) => prev.filter((entry) => entry.id !== followUp.id));
    forgetHeldFollowUp(followUp.sessionId, followUp.text);
  }

  return {
    texts: entries
      .filter((entry) => entry.sessionId === sessionId && !entry.isQueuing)
      .map((entry) => entry.text),
    hold,
    setQueuing,
    release,
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
