// The send changes this page has started on each chat, and how many are
// still out. Kept for the whole page: the composer and the "send at reset"
// menu each change the same chat's send.
const changes = new Map<string, { started: number; out: number }>();

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

/** Call as a read of the chat's send goes out; once it is back, the
 *  returned check says whether it can have missed a change from this page. */
export function startSendRead(sessionId: string) {
  const chat = changesTo(sessionId);
  const startedBefore = chat.out === 0 ? chat.started : null;
  return function canHaveMissedAChange() {
    return startedBefore === null || chat.started !== startedBefore;
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
