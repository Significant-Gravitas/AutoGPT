import type { VoiceCallCaps } from "@/app/api/__generated__/models/voiceCallCaps";

// The resume ticket of the onboarding voice call this tab holds, kept so a
// reload of the step can rejoin the call with its clock and turns. It names
// the call and how far it had got; the socket to rejoin on always comes from
// the server.

const KEY_PREFIX = "autogpt:onboarding-voice-call:";
// A ticket outlives the call cap by a minute.
const TICKET_GRACE_MS = 60_000;
const MAX_TICKET_CHARS = 2_048;
const CALL_ID = /^[0-9a-f]{32}$/;

export interface CallTicket {
  callId: string;
  ticket: string;
  caps: VoiceCallCaps;
  // When the call went live (epoch ms): its clock runs from here.
  startedAt: number;
  turnsLeft: number;
}

export function isCallId(value: unknown): value is string {
  return typeof value === "string" && CALL_ID.test(value);
}

export function isVoiceCallCaps(value: unknown): value is VoiceCallCaps {
  if (!value || typeof value !== "object") return false;
  const caps = value as Record<string, unknown>;
  return [
    caps.max_call_s,
    caps.soft_wrap_s,
    caps.max_agent_turns,
    caps.soft_wrap_turns,
    caps.idle_s,
  ].every(isCount);
}

export function saveCallTicket(userId: string, call: CallTicket) {
  try {
    sessionStorage.setItem(`${KEY_PREFIX}${userId}`, JSON.stringify(call));
  } catch {
    // A call goes on without its ticket; it just cannot be rejoined.
  }
}

// The ticket this tab kept for *callId*, or null. A ticket that cannot be
// read, or that has outlived its call, is forgotten.
export function readCallTicket(
  userId: string,
  callId: string,
): CallTicket | null {
  const key = `${KEY_PREFIX}${userId}`;
  let raw: string | null;
  try {
    raw = sessionStorage.getItem(key);
  } catch {
    return null;
  }
  if (!raw) return null;
  const kept = callTicketOf(parsedOrNull(raw));
  if (!kept || hasExpired(kept)) {
    forgetCallTicket(userId);
    return null;
  }
  return kept.callId === callId ? kept : null;
}

export function forgetCallTicket(userId: string) {
  try {
    sessionStorage.removeItem(`${KEY_PREFIX}${userId}`);
  } catch {
    // Nothing kept, nothing to forget.
  }
}

// Every ticket this tab kept, for signing out.
export function forgetCallTickets() {
  try {
    const keys = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.key(index),
    );
    keys
      .filter((key): key is string => !!key && key.startsWith(KEY_PREFIX))
      .forEach((key) => sessionStorage.removeItem(key));
  } catch {
    // Nothing kept, nothing to forget.
  }
}

function callTicketOf(value: unknown): CallTicket | null {
  if (!value || typeof value !== "object") return null;
  const kept = value as Record<string, unknown>;
  if (
    !isCallId(kept.callId) ||
    typeof kept.ticket !== "string" ||
    !kept.ticket ||
    kept.ticket.length > MAX_TICKET_CHARS ||
    !isVoiceCallCaps(kept.caps) ||
    !isCount(kept.startedAt) ||
    !isCount(kept.turnsLeft)
  ) {
    return null;
  }
  return {
    callId: kept.callId,
    ticket: kept.ticket,
    caps: kept.caps,
    startedAt: kept.startedAt,
    turnsLeft: kept.turnsLeft,
  };
}

function parsedOrNull(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function hasExpired(kept: CallTicket) {
  return (
    Date.now() >= kept.startedAt + kept.caps.max_call_s * 1000 + TICKET_GRACE_MS
  );
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
