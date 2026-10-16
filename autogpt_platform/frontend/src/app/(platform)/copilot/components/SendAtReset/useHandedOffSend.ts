import { useEffect, useRef } from "react";
import { useScheduledSendHandoff } from "./scheduledSendHandoff";

interface Actions {
  /** Sets the send, or puts the message back in the draft if it can't. */
  scheduleQuietly: (message: string) => Promise<void>;
  /** For a send a page sent and lost track of: see whether it was set. */
  recover: (message: string) => Promise<void>;
}

// Sessions this page is setting a handed-off send for right now. A send
// marked sent that isn't here was sent by a page that has since gone.
const settingSends = new Set<string>();

/** Sets the send a new-chat screen handed this session when it created it. */
export function useHandedOffSend(sessionId: string | null, actions: Actions) {
  const waiting = useScheduledSendHandoff((state) =>
    sessionId ? (state.sends[sessionId] ?? null) : null,
  );
  const latest = useRef(actions);
  // Before the handoff's effect, so it uses the actions of a committed render.
  useEffect(() => {
    latest.current = actions;
  });
  useEffect(() => {
    if (!sessionId || waiting === null || settingSends.has(sessionId)) return;
    settingSends.add(sessionId);
    const handoff = useScheduledSendHandoff.getState();
    const notSentYet = waiting.sentAt === null;
    if (notSentYet) handoff.markSent(sessionId);
    // Retrying a send a gone page may have set could add it twice.
    const settle = notSentYet
      ? latest.current.scheduleQuietly
      : latest.current.recover;
    void settle(waiting.message).finally(() => {
      settingSends.delete(sessionId);
      handoff.clear(sessionId);
    });
  }, [sessionId, waiting]);
}
