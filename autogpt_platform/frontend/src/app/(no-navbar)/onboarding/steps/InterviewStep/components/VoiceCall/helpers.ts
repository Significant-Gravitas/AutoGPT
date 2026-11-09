import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import type { VoiceCallLeg } from "@/app/api/__generated__/models/voiceCallLeg";
import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import type { VoiceCallStatusResponseStatus } from "@/app/api/__generated__/models/voiceCallStatusResponseStatus";
import {
  type CallRoomPass,
  isCallRoomPass,
  isVoiceCallCaps,
} from "@/services/onboarding/voice-call-ticket";
import { describeWait, type Refusal } from "../../helpers";
import type { CallLeg, RelayMessage } from "./protocol";

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
// A relay that refuses a start with "try again" (1013) gives the start token
// back first, or never spent it, so the same token is tried again, a few
// times. A retry the relay refuses outright (1008) is a start that failed.
export const START_RETRIES = 3;
export const START_RETRY_MS = 1_500;
// "Reconnected" stays up this long after a re-attach.
export const RESUMED_NOTICE_MS = 4_000;
// A Hang up the relay has not answered with `ended` this soon is sent again
// through the REST API's end route.
export const HANGUP_FALLBACK_MS = 5_000;
// A read the step opens on that has not answered this soon is given up, and
// the choice shows.
export const OPENING_READ_TIMEOUT_MS = 8_000;
// On the LiveKit leg the browser joins the call's room before it starts the
// call; a join LiveKit has not let through this soon takes the call's audio
// onto the socket instead.
export const ROOM_JOIN_TIMEOUT_MS = 4_000;
// Once the call ends, the browser stays in the room while Otto's last words
// play, until the relay leaves it, for at most this long.
export const ROOM_PLAYOUT_MS = 20_000;

// --- The LiveKit leg ------------------------------------------------------

// The room the start names for a call on the LiveKit leg, when it names one
// the browser can join; null for the socket leg.
export function roomPassOf(leg: VoiceCallLeg): CallRoomPass | null {
  if (leg.kind !== "livekit") return null;
  const pass = { url: leg.url, token: leg.token };
  return isCallRoomPass(pass) ? pass : null;
}

export class RoomJoinTimeout extends Error {
  constructor() {
    super("The call's room did not let the browser in in time");
    this.name = "RoomJoinTimeout";
  }
}

export type LegFallbackWhy = "timeout" | "refused" | "failed" | "relay";

// Why the browser could not join a call's room: it took too long, LiveKit
// refused it (a full plan, a token it would not take), or it failed.
export function legFallbackWhyOf(error: unknown): LegFallbackWhy {
  if (error instanceof RoomJoinTimeout) return "timeout";
  if (!error || typeof error !== "object") return "failed";
  const failure = error as Record<string, unknown>;
  if (failure.name !== "ConnectionError") return "failed";
  const status = typeof failure.status === "number" ? failure.status : 0;
  const isRefused =
    failure.reasonName === "NotAllowed" || (status >= 400 && status < 500);
  return isRefused ? "refused" : "failed";
}

// A message from the relay, read field by field, or null for anything the
// relay does not send: a frame of another shape never reaches the handlers.
export function parseRelayMessage(data: string): RelayMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") return null;
  const message = value as Record<string, unknown>;
  switch (message.type) {
    case "ready":
      return typeof message.call_id === "string" &&
        isVoiceCallCaps(message.caps) &&
        typeof message.resume === "string" &&
        isLeg(message.leg)
        ? {
            type: "ready",
            call_id: message.call_id,
            caps: message.caps,
            resume: message.resume,
            leg: message.leg,
          }
        : null;
    case "leg":
      return isLeg(message.leg) ? { type: "leg", leg: message.leg } : null;
    case "user_text":
    case "agent_text":
      return typeof message.text === "string"
        ? { type: message.type, text: message.text }
        : null;
    case "interrupted":
      return { type: "interrupted" };
    case "turn_complete":
      return isCount(message.turn) && isCount(message.turns_left)
        ? {
            type: "turn_complete",
            turn: message.turn,
            turns_left: message.turns_left,
          }
        : null;
    case "wrap_up":
      return typeof message.reason === "string"
        ? { type: "wrap_up", reason: message.reason }
        : null;
    case "resumed":
      return isCount(message.gap_s)
        ? { type: "resumed", gap_s: message.gap_s }
        : null;
    case "ended":
      return typeof message.reason === "string" && isCount(message.elapsed_s)
        ? {
            type: "ended",
            reason: message.reason,
            elapsed_s: message.elapsed_s,
          }
        : null;
    case "error":
      return typeof message.error === "string"
        ? {
            type: "error",
            error: message.error,
            call_id:
              typeof message.call_id === "string" ? message.call_id : null,
          }
        : null;
    case "pong":
      return {
        type: "pong",
        id:
          typeof message.id === "number" || typeof message.id === "string"
            ? message.id
            : null,
      };
    default:
      return null;
  }
}

function isLeg(value: unknown): value is CallLeg {
  return value === "websocket" || value === "livekit";
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

// The relay's call socket as the server named it: a secure socket, or a plain
// one only for a page served over plain HTTP (a local install).
export function isRelaySocketUrl(value: unknown): value is string {
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
  // The person's finish of the call waits until the server can take it:
  // the call's words, or a later call of theirs, are not settled yet.
  | "finishing"
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
  // This tab had the call until another tab or device rejoined it.
  wentElsewhere: boolean;
}

interface CallStatus {
  phase: CallPhase;
  isWrappingUp: boolean;
  wasResumed: boolean;
  isHangingUp: boolean;
  // LiveKit is getting the call's audio back after the network changed.
  isAudioReconnecting: boolean;
}

// The line under Otto while a call runs or its words are being saved. After
// a drop nothing mentions typing until the words have an answer.
export function callStatusCopy({
  phase,
  isWrappingUp,
  wasResumed,
  isHangingUp,
  isAudioReconnecting,
}: CallStatus) {
  switch (phase) {
    case "starting":
      return "Calling Otto…";
    case "live":
      if (isHangingUp) return "Hanging up…";
      if (isAudioReconnecting) return "Reconnecting…";
      if (wasResumed) return "Reconnected";
      if (isWrappingUp) return "Otto is wrapping up";
      return "You're on the call with Otto";
    case "reconnecting":
      return "Reconnecting…";
    case "wrapping_up":
      return "Wrapping up…";
    case "saving":
      return "We lost the call. Saving what you said…";
    case "finishing":
      return "Saving what you said…";
    case "ending_call":
      return "Ending the call…";
    default:
      return null;
  }
}

// --- How a call ended -----------------------------------------------------

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

// The call is running, or its words are still on their way to a take: being
// transcribed, or handed off (the call routes report a call owed a hand-off
// as `handing_off`). `transcribed` alone means a confirm is awaited.
const UNSETTLED = new Set<VoiceCallStatusResponseStatus>([
  "connecting",
  "live",
  "ended",
  "transcribing",
  "handing_off",
]);

const SUPERSEDED = "superseded";
// Nothing the person said was heard on the call.
const NOTHING_HEARD = "no_usable_speech";
// Words were heard, and judged too few to set up from.
const NOT_ENOUGH_WORDS = "insufficient_content";
// The pipeline never ruled on words the call kept: the finish route hands
// them off again. Any other failure kept nothing to hand off, and is ours:
// the recording or its transcript could not be made
// (`recording_unavailable`, `transcription_unavailable`), or a failure the
// step does not know.
const HANDED_OFF_AGAIN = new Set([
  "quality_check_unavailable",
  "rate_limited",
  "personalization_unavailable",
]);

export type CallEnding =
  // "Meet your team". `path` is the one the hand-off decided; null when the
  // click must finish the call first.
  | { kind: "team"; path: "A" | null }
  // Cut off with the words saved: "Meet your team" (a finish of this call),
  // "Call again" and "Type instead".
  | { kind: "lost"; reason: string | null }
  // Nothing the person said was heard: "Call again" and "Type instead".
  | { kind: "no_words" }
  | { kind: "insufficient" }
  // The pipeline never ruled on the words: "Try again" through the finish
  // route, beside the other ways on after a cut-off call.
  | { kind: "retry"; cutOff: boolean; reason: string | null }
  // Something went wrong on our side saving the call, so it kept nothing to
  // hand off: "Call again" and "Type instead".
  | { kind: "unsaved"; reason: string | null }
  // Nothing is left to read of the call.
  | { kind: "missing" };

export function ranToClose(reason: string | null | undefined) {
  return !!reason && RAN_TO_CLOSE.has(reason);
}

export function isSettled(call: VoiceCallStatusResponse) {
  return !UNSETTLED.has(call.status);
}

// How the step answers a call it reads. The record's word on who owns the
// interview comes first: a call the person's typing superseded is no screen,
// so the typed interview opens. Then, once the words have an answer, the end
// reason and the status decide the call's ending; null while the words are
// on their way. `taken` says the server named this call as the one that took
// the interview.
export function answerOf(
  call: VoiceCallStatusResponse,
  { taken = false }: { taken?: boolean } = {},
): CallEnding | "typed" | null {
  if (call.interview === "superseded") return "typed";
  if (!isSettled(call)) return null;
  const closed =
    taken || call.interview === "taken" || ranToClose(call.end_reason);
  const reason = call.end_reason ?? null;
  if (call.status === "failed") {
    const failure = failureOf(call.hand_off?.error_code ?? null);
    if (failure === "no_words") return { kind: "no_words" };
    if (failure === "insufficient") return { kind: "insufficient" };
    if (failure === "retry") return { kind: "retry", cutOff: !closed, reason };
    return { kind: "unsaved", reason };
  }
  if (closed && call.status === "handed_off") {
    const outcome = call.hand_off ? voiceFinishOutcome(call.hand_off) : null;
    // Not the person's typing, read above: another take that beat this one
    // owns the interview, and leads on as an accepted take does.
    const leadsOn = outcome === "A" || outcome === "superseded";
    return { kind: "team", path: leadsOn ? "A" : null };
  }
  if (closed && call.status === "transcribed")
    return { kind: "team", path: null };
  return { kind: "lost", reason };
}

// What a finish of a call answers, read for a call: an accepted take leads
// on to the team; `superseded` says another take owns the interview, which
// the call's record tells apart (the person's typing, or a take that leads
// on); a call with no words, or too few, keeps the interview open for
// another call or for typing; a hand-off nobody ruled on is tried again
// (null).
export function voiceFinishOutcome(
  response: FinalizeResponse,
): "A" | "superseded" | "no_words" | "insufficient" | "unsaved" | null {
  if (response.status !== "failed") return "A";
  const code = response.error_code ?? null;
  if (code === SUPERSEDED) return "superseded";
  const failure = failureOf(code);
  return failure === "retry" ? null : failure;
}

// A failed hand-off, read for what the person can do next: nothing heard,
// or a verdict that the words were too few; words kept to hand off again;
// or nothing kept, through a failure of ours.
function failureOf(
  code: string | null,
): "no_words" | "insufficient" | "retry" | "unsaved" {
  if (code === NOTHING_HEARD) return "no_words";
  if (code === NOT_ENOUGH_WORDS) return "insufficient";
  if (code === null || HANDED_OFF_AGAIN.has(code)) return "retry";
  return "unsaved";
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
  superseded: "The call ended before Otto could finish.",
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

// Seconds from *start* (epoch ms) to *end*, to a tenth, as the relay counts
// a call's length.
export function secondsBetween(start: number, end: number) {
  return Math.round(Math.max(0, end - start) / 100) / 10;
}

export function timeLeftCopy(msLeft: number) {
  const seconds = Math.max(0, Math.round(msLeft / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} left`;
}

export function turnsLeftCopy(turnsLeft: number) {
  if (turnsLeft <= 0) return "Wrapping up";
  return turnsLeft === 1 ? "1 turn left" : `${turnsLeft} turns left`;
}
