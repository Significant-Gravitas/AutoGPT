import { holdsMessage } from "./helpers";

// The send changes this page has started on each chat, and how many are
// still out. Kept for the whole page: the composer and the "send at reset"
// menu each change the same chat's send.
const changes = new Map<string, { started: number; out: number }>();
// The waits on each chat's send, each told the text of a send this page
// cancels while it waits.
const cancelWatches = new Map<string, Set<(cancelled: string) => void>>();

/** Mutation callbacks that count each change to a chat's send. */
export const countSendChanges = {
  onMutate({ sessionId }: { sessionId: string }) {
    const chat = changesTo(sessionId);
    chat.started += 1;
    chat.out += 1;
  },
  onSettled(_data: unknown, _error: unknown, variables: { sessionId: string }) {
    changesTo(variables.sessionId).out -= 1;
  },
};

/** Call as this page cancels the chat's send, with the text it cancels. */
export function noteCancel(sessionId: string, cancelled: string) {
  cancelWatches.get(sessionId)?.forEach((watch) => watch(cancelled));
}

/** Call as a read of the chat's send goes out; once it is back, the
 *  returned check says whether it can have missed a change from this page. */
export function startSendRead(sessionId: string) {
  const chat = changesTo(sessionId);
  const startedBefore = chat.out === 0 ? chat.started : null;
  return function canHaveMissedAChange() {
    return startedBefore === null || chat.started !== startedBefore;
  };
}

/** Call as a wait on the chat's send begins; the returned check ends the
 *  wait and says whether this page has since cancelled a send holding
 *  `message`. */
export function startCancelWatch(sessionId: string, message: string) {
  let cancelled = false;
  function watch(text: string) {
    if (holdsMessage(text, message)) cancelled = true;
  }
  const watches = cancelWatches.get(sessionId) ?? new Set();
  watches.add(watch);
  cancelWatches.set(sessionId, watches);
  return function cancelledSince() {
    watches.delete(watch);
    if (watches.size === 0) cancelWatches.delete(sessionId);
    return cancelled;
  };
}

function changesTo(sessionId: string) {
  let chat = changes.get(sessionId);
  if (!chat) {
    chat = { started: 0, out: 0 };
    changes.set(sessionId, chat);
  }
  return chat;
}
