import type { VoiceCallCaps } from "@/app/api/__generated__/models/voiceCallCaps";

// The resume ticket of the onboarding voice call this tab holds, kept so a
// reload of the step can rejoin the call with its clock and turns. It names
// the call and how far it had got, and on the LiveKit leg the call's room
// with this browser's token for it; the socket to rejoin on always comes
// from the server.

const KEY_PREFIX = "autogpt:onboarding-voice-call:";
// A ticket outlives the call cap by a minute.
const TICKET_GRACE_MS = 60_000;
const MAX_TICKET_CHARS = 2_048;
const MAX_ROOM_TOKEN_CHARS = 4_096;
const CALL_ID = /^[0-9a-f]{32}$/;

// A call's LiveKit room, and this browser's token for it.
export interface CallRoomPass {
  url: string;
  token: string;
}

export interface CallTicket {
  callId: string;
  ticket: string;
  caps: VoiceCallCaps;
  // When the call went live (epoch ms): its clock runs from here.
  startedAt: number;
  turnsLeft: number;
  // On the LiveKit leg, the room a rejoin joins again; null on the socket.
  room: CallRoomPass | null;
}

export function isCallId(value: unknown): value is string {
  return typeof value === "string" && CALL_ID.test(value);
}

// A room the server named: a LiveKit project's secure socket, or a plain one
// only for a page served over plain HTTP (a local install), and a token.
export function isCallRoomPass(value: unknown): value is CallRoomPass {
  if (!value || typeof value !== "object") return false;
  const room = value as Record<string, unknown>;
  return (
    isRoomUrl(room.url) &&
    typeof room.token === "string" &&
    room.token.length > 0 &&
    room.token.length <= MAX_ROOM_TOKEN_CHARS
  );
}

function isRoomUrl(value: unknown) {
  if (typeof value !== "string") return false;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return false;
  }
  if (url.protocol === "wss:") return true;
  return url.protocol === "ws:" && window.location.protocol === "http:";
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

// Every ticket this tab kept, for signing out; with *keepUserId*, every one
// but that user's, for an account taking over the tab.
export function forgetCallTickets(keepUserId?: string) {
  const kept = keepUserId === undefined ? null : `${KEY_PREFIX}${keepUserId}`;
  try {
    const keys = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.key(index),
    );
    keys
      .filter(
        (key): key is string =>
          !!key && key.startsWith(KEY_PREFIX) && key !== kept,
      )
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
  // A call kept with a room is on the LiveKit leg: one whose room cannot be
  // read cannot be rejoined, since its audio is in that room.
  const room = kept.room ?? null;
  if (room !== null && !isCallRoomPass(room)) return null;
  return {
    callId: kept.callId,
    ticket: kept.ticket,
    caps: kept.caps,
    startedAt: kept.startedAt,
    turnsLeft: kept.turnsLeft,
    room: isCallRoomPass(room) ? { url: room.url, token: room.token } : null,
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
