import { useEffect, useRef } from "react";

// How long a switch of the composer's recipient may take before the task
// gives up instead of going to whoever is selected.
const RECIPIENT_SWITCH_MS = 3000;

interface Args {
  recipientID: string | null;
  selectRecipient: (id: string | null) => void;
  /** The page can start a task now: false while it is still resolving the
   *  new recipient, or the composer is locked. */
  canSend: boolean;
}

interface Pending {
  id: string | null;
  send: () => void;
  timer: ReturnType<typeof setTimeout>;
}

// Sends a task to its owner the way the recipient picker would: the
// composer's recipient becomes the owner (the ?expertId= param), and the
// task goes out once the page has picked that up and can send, so the chat
// is created for the owner. `restore` puts back the recipient the user had
// when a task does not start after all.
export function useTaskRecipient({
  recipientID,
  selectRecipient,
  canSend,
}: Args) {
  const pending = useRef<Pending | null>(null);
  const previous = useRef<string | null>(null);

  useEffect(() => {
    const next = pending.current;
    if (!next || next.id !== recipientID || !canSend) return;
    clearTimeout(next.timer);
    pending.current = null;
    next.send();
  }, [recipientID, canSend]);

  useEffect(() => {
    return () => {
      if (pending.current) clearTimeout(pending.current.timer);
    };
  }, []);

  function sendTo(id: string | null, send: () => void, onGiveUp: () => void) {
    previous.current = recipientID;
    if (id === recipientID && canSend) {
      send();
      return;
    }
    if (pending.current) clearTimeout(pending.current.timer);
    const timer = setTimeout(() => {
      if (pending.current?.timer !== timer) return;
      pending.current = null;
      onGiveUp();
    }, RECIPIENT_SWITCH_MS);
    pending.current = { id, send, timer };
    if (id !== recipientID) selectRecipient(id);
  }

  function restore() {
    if (previous.current !== recipientID) selectRecipient(previous.current);
  }

  return { sendTo, restore };
}
