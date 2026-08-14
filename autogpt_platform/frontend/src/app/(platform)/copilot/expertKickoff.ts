import type { UIMessage } from "ai";

// The single hidden user message sent on an expert's first day. The identity
// and installed-workflow context blocks are injected server-side, so this only
// has to tell the expert what to do with them.
const KICKOFF_MESSAGE =
  "You were just hired. Introduce yourself in 2-3 sentences in your voice, " +
  "state your day-one job from your installed workflows, then either start " +
  "your first bundled workflow with run_agent or, if access/credentials are " +
  "missing, ask for exactly the one connection the day-one job needs and why. " +
  "Never pretend a run succeeded.";

export function buildKickoffMessage(): string {
  return KICKOFF_MESSAGE;
}

export function kickoffStorageKey(expertId: string): string {
  return `expert-kickoff-${expertId}`;
}

export function hasKickedOff(expertId: string): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(kickoffStorageKey(expertId)) !== null;
  } catch {
    return false;
  }
}

export function markKickedOff(expertId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(kickoffStorageKey(expertId), "1");
  } catch {
    // A blocked or full localStorage only costs us the once-per-expert latch,
    // and the empty-sessions check still guards against a duplicate kickoff.
  }
}

function messageText(message: UIMessage): string {
  return message.parts
    .filter(
      (part): part is Extract<typeof part, { type: "text" }> =>
        part.type === "text",
    )
    .map((part) => part.text)
    .join("")
    .trim();
}

// The kickoff prompt is a user message that only exists to provoke the
// expert's introduction, so it must never render — not on the optimistic send
// and not after a reload rehydrates it from the backend.
export function isKickoffMessage(message: UIMessage): boolean {
  return message.role === "user" && messageText(message) === KICKOFF_MESSAGE;
}

export function stripKickoffMessages(messages: UIMessage[]): UIMessage[] {
  return messages.filter((message) => !isKickoffMessage(message));
}
