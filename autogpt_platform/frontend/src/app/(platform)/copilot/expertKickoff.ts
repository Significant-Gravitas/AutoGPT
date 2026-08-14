import type { UIMessage } from "ai";

// Structural marker prefixed to the kickoff user message. Hiding keys on this
// marker (never on prose content), the stream transport derives the
// deterministic dedup message id from it, and it round-trips through backend
// persistence — `StreamChatRequest` carries no metadata field for user
// messages, so the marker IS the metadata channel available to the frontend.
const KICKOFF_MARKER_PREFIX = "[[EXPERT_KICKOFF:";

// The single hidden user message sent on an expert's first day. The identity
// and installed-workflow context blocks are injected server-side, so this only
// has to tell the expert what to do with them.
const KICKOFF_PROMPT =
  "You were just hired. Introduce yourself in 2-3 sentences in your voice, " +
  "state your day-one job from your installed workflows, then either start " +
  "your first bundled workflow with run_agent or, if access/credentials are " +
  "missing, ask for exactly the one connection the day-one job needs and why. " +
  "Never pretend a run succeeded.";

export function buildKickoffMessage(expertId: string): string {
  return `${KICKOFF_MARKER_PREFIX}${expertId}]]\n\n${KICKOFF_PROMPT}`;
}

export function isKickoffText(text: string): boolean {
  return text.startsWith(KICKOFF_MARKER_PREFIX);
}

function firstTextPart(message: UIMessage): string | null {
  const part = message.parts.find(
    (candidate): candidate is Extract<typeof candidate, { type: "text" }> =>
      candidate.type === "text",
  );
  return part?.text ?? null;
}

export function isKickoffMessage(message: UIMessage): boolean {
  if (message.role !== "user") return false;
  const text = firstTextPart(message);
  return text !== null && isKickoffText(text);
}

export function stripKickoffMessages<T extends UIMessage>(messages: T[]): T[] {
  return messages.filter((message) => !isKickoffMessage(message));
}

// Deterministic `message_id` for a kickoff send. The id becomes the persisted
// ChatMessage PK, so Postgres' unique constraint is the atomic server-side
// dedup: two tabs racing the same first kickoff both derive attempt 0, one
// INSERT wins, and the loser short-circuits to subscribe-only — the turn (and
// its workflow side effects) fires exactly once. A genuine retry has the
// failed kickoff in its history, derives attempt 1, and is not dead-ended.
export function deriveKickoffMessageId(messages: UIMessage[]): string | null {
  const last = messages[messages.length - 1];
  if (!last || !isKickoffMessage(last)) return null;
  const text = firstTextPart(last) ?? "";
  const expertId = text
    .slice(KICKOFF_MARKER_PREFIX.length, text.indexOf("]]"))
    .trim();
  if (!expertId) return null;
  const attempt = messages.slice(0, -1).filter(isKickoffMessage).length;
  return `expert-kickoff-${expertId}-${attempt}`.slice(0, 64);
}

// --- once-per-expert latch -------------------------------------------------
//
// localStorage is an optimization, never the correctness boundary (that is
// the deterministic message id above). Three states:
//   absent          → idle: kickoff may fire
//   "pending:<ts>"  → another tab (or this one) is mid-kickoff; skip. Expires
//                     so a crashed tab can't consume the kickoff forever.
//   "done"          → the kickoff send was accepted; never fire again.
// Legacy value "1" (pre-state-machine) reads as done.

const PENDING_TTL_MS = 2 * 60 * 1000;

export type KickoffStatus = "idle" | "pending" | "done";

export function kickoffStorageKey(expertId: string): string {
  return `expert-kickoff-${expertId}`;
}

export function getKickoffStatus(expertId: string): KickoffStatus {
  if (typeof window === "undefined") return "idle";
  try {
    const value = window.localStorage.getItem(kickoffStorageKey(expertId));
    if (value === null) return "idle";
    if (value === "done" || value === "1") return "done";
    if (value.startsWith("pending:")) {
      const startedAt = Number(value.slice("pending:".length));
      if (Number.isFinite(startedAt) && Date.now() - startedAt < PENDING_TTL_MS)
        return "pending";
      return "idle";
    }
    return "idle";
  } catch {
    return "idle";
  }
}

export function markKickoffPending(expertId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      kickoffStorageKey(expertId),
      `pending:${Date.now()}`,
    );
  } catch {
    // A blocked localStorage only weakens the cross-tab hint; the
    // deterministic message id still guarantees a single kickoff turn.
  }
}

export function markKickoffDone(expertId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(kickoffStorageKey(expertId), "done");
  } catch {
    // Same rationale as markKickoffPending.
  }
}

export function clearKickoffPending(expertId: string): void {
  if (typeof window === "undefined") return;
  try {
    const value = window.localStorage.getItem(kickoffStorageKey(expertId));
    if (value?.startsWith("pending:")) {
      window.localStorage.removeItem(kickoffStorageKey(expertId));
    }
  } catch {
    // Same rationale as markKickoffPending.
  }
}
