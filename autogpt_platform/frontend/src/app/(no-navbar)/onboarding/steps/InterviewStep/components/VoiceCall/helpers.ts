import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import type { VoiceCallCaps } from "@/app/api/__generated__/models/voiceCallCaps";
import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import { describeWait, type Refusal } from "../../helpers";
import type { RelayMessage, RelayMessageType } from "./protocol";

// A dropped socket may re-attach to its call for this long, as the relay
// holds the call for it, trying this often.
export const REATTACH_WINDOW_MS = 15_000;
export const REATTACH_RETRY_MS = 500;
// Once a call is over, what became of its words is read this often for this
// long; after that the person may type instead while it is still read, more
// slowly.
export const ENDING_POLL_MS = 2_000;
export const ENDING_WINDOW_MS = 30_000;
export const ENDING_SLOW_POLL_MS = 5_000;
// A call someone asked to end leaves the live state within a second or so.
export const END_POLL_MS = 1_000;
export const END_WAIT_MS = 10_000;
// A relay that asks the browser to try again (1013) is retried with the same
// start token, which stays good for that.
export const START_RETRIES = 3;
export const START_RETRY_MS = 1_500;
// "Reconnected" stays up this long after a re-attach.
export const RESUMED_NOTICE_MS = 4_000;

const RELAY_MESSAGE_TYPES = new Set<RelayMessageType>([
  "ready",
  "leg",
  "user_text",
  "agent_text",
  "interrupted",
  "turn_complete",
  "wrap_up",
  "resumed",
  "ended",
  "error",
  "pong",
]);

export function parseRelayMessage(data: string): RelayMessage | null {
  let message: unknown;
  try {
    message = JSON.parse(data);
  } catch {
    return null;
  }
  if (!message || typeof message !== "object") return null;
  const type = (message as { type?: unknown }).type;
  return typeof type === "string" &&
    RELAY_MESSAGE_TYPES.has(type as RelayMessageType)
    ? (message as RelayMessage)
    : null;
}

// --- The call's screen ------------------------------------------------------

export type CallPhase =
  // Reading the person's latest call, for a step that opens again.
  | "checking"
  // The choice: "Talk to Otto" or "Type instead".
  | "idle"
  // The microphone, the call's start and its socket, before `ready`.
  | "starting"
  | "live"
  // A dropped socket re-attaching to its call, or a rejoin.
  | "reconnecting"
  // The call ended; what became of its words is being read.
  | "wrapping_up"
  // The call was lost; what became of its words is being read.
  | "saving"
  // Ending a live call before what the person chose next.
  | "ending_call"
  // A live call holds the interview: rejoin it or end it.
  | "held"
  | "ended";

export interface HeldCall {
  callId: string;
  canRejoin: boolean;
  // After "End that call": the typed start or the new call the live one
  // refused, or the ended call's own ending.
  then: "type" | "call" | "show";
}

interface CallStatus {
  phase: CallPhase;
  isWrappingUp: boolean;
  wasResumed: boolean;
  isHangingUp: boolean;
}

// The line under Otto while a call runs or its words are being saved. After
// a drop nothing mentions typing until the words have an answer.
export function callStatusCopy({
  phase,
  isWrappingUp,
  wasResumed,
  isHangingUp,
}: CallStatus) {
  switch (phase) {
    case "starting":
      return "Calling Otto…";
    case "live":
      if (isHangingUp) return "Hanging up…";
      if (wasResumed) return "Reconnected";
      if (isWrappingUp) return "Otto is wrapping up";
      return "You're on the call with Otto";
    case "reconnecting":
      return "Reconnecting…";
    case "wrapping_up":
      return "Wrapping up…";
    case "saving":
      return "We lost the call. Saving what you said…";
    case "ending_call":
      return "Ending the call…";
    default:
      return null;
  }
}

// --- How a call ended (research/17 §2.2 step 9, §2.6) ---------------------

// A call that ran to its close: Otto's goodbye, the server's caps, the scope
// rule or the person's Hang up. Its accepted words take the interview. Every
// other end cut the call off, which leaves the interview open.
const RAN_TO_CLOSE = new Set([
  "finished",
  "wall_clock",
  "turn_cap",
  "spend_cap",
  "scope",
  "user_hangup",
]);

// The call is running, or its words are still being made into a take.
const UNSETTLED = new Set(["connecting", "live", "ended", "transcribing"]);

const SUPERSEDED = "superseded";
const VERDICTS_ON_THE_WORDS = new Set([
  "no_usable_speech",
  "insufficient_content",
]);

export type CallEnding =
  // "Meet your team". `path` is the one the hand-off decided; null when the
  // click must finish the call first.
  | { kind: "team"; path: "A" | null }
  // Cut off with the words saved: "Meet your team" (a finish of this call),
  // "Call again" and "Type instead".
  | { kind: "lost"; reason: string | null }
  | { kind: "insufficient" }
  // The pipeline never ruled on the words: "Try again" through the finish
  // route, beside the other ways on after a cut-off call.
  | { kind: "retry"; cutOff: boolean; reason: string | null }
  // Nothing is left to read of the call.
  | { kind: "missing" };

export function ranToClose(reason: string | null | undefined) {
  return !!reason && RAN_TO_CLOSE.has(reason);
}

export function isSettled(call: VoiceCallStatusResponse) {
  return !UNSETTLED.has(call.status);
}

// Who owns the interview, as the call's record says: its words took it, or
// a take the person made after it did (research/17 §2.6). Absent while
// neither has happened.
export type CallInterview = "taken" | "superseded";

export function interviewOf(
  call: VoiceCallStatusResponse,
): CallInterview | null {
  const value = (call as { interview?: unknown }).interview;
  return value === "taken" || value === "superseded" ? value : null;
}

// How the step answers a call it reads. The record's word on the interview
// comes first: a call a later take superseded is no screen, so the typed
// interview opens. Then, once the words have an answer, the end reason and
// the status decide the call's ending; null while the words are on their
// way. `taken` says the server named this call as the one that took the
// interview.
export function answerOf(
  call: VoiceCallStatusResponse,
  { taken = false }: { taken?: boolean } = {},
): CallEnding | "typed" | null {
  const interview = interviewOf(call);
  if (interview === "superseded") return "typed";
  if (!isSettled(call)) return null;
  const closed = taken || interview === "taken" || ranToClose(call.end_reason);
  const reason = call.end_reason ?? null;
  if (call.status === "failed") {
    if (VERDICTS_ON_THE_WORDS.has(call.hand_off?.error_code ?? "")) {
      return { kind: "insufficient" };
    }
    return { kind: "retry", cutOff: !closed, reason };
  }
  if (closed && call.status === "handed_off") {
    const path = call.hand_off ? voiceFinishOutcome(call.hand_off) : null;
    return { kind: "team", path: path === "A" ? "A" : null };
  }
  if (closed && call.status === "transcribed")
    return { kind: "team", path: null };
  return { kind: "lost", reason };
}

// What a finish of a call answers, read for a call: an accepted take, or one
// a newer take owns, leads on to the team; a verdict on the words keeps the
// interview open for another call or for typing; anything else is tried
// again.
export function voiceFinishOutcome(
  response: FinalizeResponse,
): "A" | "insufficient" | null {
  if (response.status !== "failed") return "A";
  const code = response.error_code ?? "";
  if (code === SUPERSEDED) return "A";
  if (VERDICTS_ON_THE_WORDS.has(code)) return "insufficient";
  return null;
}

const ENDED_COPY: Record<string, string> = {
  finished: "That's everything Otto needed.",
  wall_clock: "That's the time we had for this call.",
  turn_cap: "That's all the questions for this call.",
  spend_cap: "That's the time we had for this call.",
  scope: "This call is only for learning about your work, so Otto ended it.",
  user_hangup: "You ended the call.",
  client_gone: "The connection dropped.",
  idle: "We didn't hear anything for a while, so the call ended.",
  vendor_error: "Otto's line dropped.",
  server_restart: "Our server restarted during the call.",
  superseded: "The call was ended from another tab or device.",
  crashed: "Our server stopped during the call.",
};

export function endedCopy(reason: string | null | undefined) {
  return (reason && ENDED_COPY[reason]) || "The call ended.";
}

// --- Starting a call --------------------------------------------------------

export type CallStartFailure =
  | { kind: "busy"; retryAfterSeconds: number | null }
  | { kind: "unavailable"; retryAfterSeconds: number | null }
  | { kind: "failed" };

export function callStartFailureOf(refusal: Refusal): CallStartFailure {
  const retryAfterSeconds = refusal.retryAfterSeconds;
  if (refusal.status === 429) return { kind: "busy", retryAfterSeconds };
  if (refusal.status === 503) return { kind: "unavailable", retryAfterSeconds };
  return { kind: "failed" };
}

export function callStartFailureCopy(failure: CallStartFailure) {
  if (failure.kind === "busy") {
    const wait = failure.retryAfterSeconds
      ? `in ${describeWait(failure.retryAfterSeconds)}`
      : "a little later";
    return `You've made a few calls already. You can call again ${wait}, or type instead.`;
  }
  if (failure.kind === "unavailable") {
    return "Otto can't take calls right now. You can type instead, or try again in a moment.";
  }
  return "We couldn't start the call. Try again, or type instead.";
}

export type MicrophoneRefusal = "blocked" | "unavailable";

// A refused permission is the person's or the browser's settings; anything
// else (no microphone, one another app holds) is the device.
export function microphoneRefusalOf(error: unknown): MicrophoneRefusal {
  const name =
    typeof error === "object" && error !== null && "name" in error
      ? String((error as { name: unknown }).name)
      : "";
  return name === "NotAllowedError" || name === "SecurityError"
    ? "blocked"
    : "unavailable";
}

// --- The call's resume ticket, kept for a reload in this tab ---------------

const STORED_CALL_KEY_PREFIX = "autogpt:onboarding-voice-call:";
// A ticket outlives the call cap by a minute.
const TICKET_GRACE_MS = 60_000;

export interface CallTicket {
  callId: string;
  relayUrl: string;
  ticket: string;
  caps: VoiceCallCaps;
  // When the call went live (epoch ms): its clock runs from here.
  startedAt: number;
  turnsLeft: number;
}

export function saveCallTicket(userId: string, call: CallTicket) {
  try {
    sessionStorage.setItem(
      `${STORED_CALL_KEY_PREFIX}${userId}`,
      JSON.stringify(call),
    );
  } catch {
    // A call goes on without its ticket; it just cannot be rejoined.
  }
}

export function readCallTicket(
  userId: string,
  callId: string,
): CallTicket | null {
  let call: CallTicket;
  try {
    const raw = sessionStorage.getItem(`${STORED_CALL_KEY_PREFIX}${userId}`);
    if (!raw) return null;
    call = JSON.parse(raw) as CallTicket;
  } catch {
    return null;
  }
  if (
    call?.callId !== callId ||
    typeof call.ticket !== "string" ||
    typeof call.relayUrl !== "string"
  ) {
    return null;
  }
  const expiresAt =
    call.startedAt + call.caps.max_call_s * 1000 + TICKET_GRACE_MS;
  return Date.now() < expiresAt ? call : null;
}

export function forgetCallTicket(userId: string) {
  try {
    sessionStorage.removeItem(`${STORED_CALL_KEY_PREFIX}${userId}`);
  } catch {
    // Nothing kept, nothing to forget.
  }
}

// --- Captions ---------------------------------------------------------------

export interface CaptionLine {
  id: number;
  who: "otto" | "you";
  text: string;
}

// Adds a fragment of a live caption, to the line being spoken or as a new
// line when the speaker changes or a turn has ended.
export function withCaption(
  lines: CaptionLine[],
  who: CaptionLine["who"],
  text: string,
  isNewTurn: boolean,
): CaptionLine[] {
  const last = lines.at(-1);
  if (last && last.who === who && !isNewTurn) {
    return [...lines.slice(0, -1), { ...last, text: last.text + text }];
  }
  return [...lines, { id: (last?.id ?? 0) + 1, who, text: text.trimStart() }];
}

// --- The clock and the turns ------------------------------------------------

export function timeLeftCopy(msLeft: number) {
  const seconds = Math.max(0, Math.round(msLeft / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} left`;
}

export function turnsLeftCopy(turnsLeft: number) {
  if (turnsLeft <= 0) return "Wrapping up";
  return turnsLeft === 1 ? "1 turn left" : `${turnsLeft} turns left`;
}
