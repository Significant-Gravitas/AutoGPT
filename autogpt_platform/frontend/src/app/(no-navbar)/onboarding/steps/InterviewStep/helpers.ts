import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { UIMessage } from "ai";

// The stages the backend records once the interview takes no more turns.
const ENDED_STAGES = new Set(["closed", "handed_off"]);

const SUPERSEDED = "superseded";
const VERDICTS_ON_THE_WORDS = new Set([
  "no_usable_speech",
  "insufficient_content",
]);

// What the chat routes answer a turn sent after the interview is over.
const INTERVIEW_CLOSED = "onboarding_interview_closed";
// What the typed start and the call start answer while a voice call of the
// person's is live.
const CALL_IN_PROGRESS = "call_in_progress";

const TYPING_KEY_PREFIX = "autogpt:onboarding-interview-typing:";

// After each turn the stage is read again, because the reply cap and the
// scope guard close the interview a moment after the turn ends.
export const STAGE_READ_INTERVAL_MS = 2_000;
export const STAGE_READ_WINDOW_MS = 10_000;

export type StartFailure =
  | { kind: "busy"; retryAfterSeconds: number | null }
  | { kind: "unavailable" }
  | { kind: "failed" };

export function isEndedStage(stage: string | null | undefined) {
  return !!stage && ENDED_STAGES.has(stage);
}

// The server writes a hidden kickoff so Otto speaks first; it is not
// something the person said.
export function withoutKickoff<T extends UIMessage>(messages: T[]): T[] {
  return messages.filter((message) => !isHiddenMessage(message));
}

export function hasAnswered(messages: UIMessage[]) {
  return withoutKickoff(messages).some((message) => message.role === "user");
}

export function isInterviewClosedError(error: Error | undefined) {
  return !!error?.message?.includes(INTERVIEW_CLOSED);
}

export function startFailureOf(error: unknown): StartFailure {
  if (error instanceof ApiError) {
    if (error.status === 429) {
      return { kind: "busy", retryAfterSeconds: retryAfterSecondsOf(error) };
    }
    if (error.status === 503) return { kind: "unavailable" };
  }
  return { kind: "failed" };
}

export function finishFailureOf(error: unknown): StartFailure {
  if (error instanceof ApiError && error.status === 429) {
    return { kind: "busy", retryAfterSeconds: retryAfterSecondsOf(error) };
  }
  return { kind: "failed" };
}

export function finishFailureCopy(failure: StartFailure) {
  return failure.kind === "busy"
    ? startFailureCopy(failure)
    : "That didn't go through. Try again.";
}

// The path the take the finish answered with puts the person on, or null
// when the pipeline never ruled on their words and the click should be
// tried again.
export function finishPathOf(response: FinalizeResponse): "A" | "B" | null {
  if (response.status !== "failed") return "A";
  const code = response.error_code ?? "";
  if (code === SUPERSEDED) return "A";
  if (VERDICTS_ON_THE_WORDS.has(code)) return "B";
  return null;
}

export function startFailureCopy(failure: StartFailure) {
  if (failure.kind === "busy") {
    const wait = failure.retryAfterSeconds
      ? `in ${describeWait(failure.retryAfterSeconds)}`
      : "in a little while";
    return `You've sent a lot of messages in a short time. Your progress is saved; try again ${wait}.`;
  }
  if (failure.kind === "unavailable") {
    return "Otto can't chat right now. Your progress is saved; try again in a moment.";
  }
  return "We couldn't start your interview. Try again.";
}

export function describeWait(seconds: number) {
  const minutes = Math.ceil(seconds / 60);
  if (minutes <= 1) return "a minute";
  if (minutes < 60) return `${minutes} minutes`;
  const hours = Math.ceil(minutes / 60);
  return hours === 1 ? "an hour" : `${hours} hours`;
}

// A person has one interview, typed or spoken. A start refused because a
// voice call holds it names that call: a call still live, or one whose words
// took the interview (no call when the typed interview itself is over).
export type HeldBy =
  | { kind: "call_in_progress"; callId: string }
  | { kind: "closed"; callId: string | null };

export interface Refusal {
  status: number | null;
  retryAfterSeconds: number | null;
  heldBy: HeldBy | null;
}

export function refusalOf(error: unknown): Refusal {
  if (!(error instanceof ApiError)) {
    return { status: null, retryAfterSeconds: null, heldBy: null };
  }
  return {
    status: error.status,
    retryAfterSeconds: retryAfterSecondsOf(error),
    heldBy: error.status === 409 ? heldByOf(error.response?.detail) : null,
  };
}

// A typed turn refused because a voice call holds the interview. The chat
// transport reports a refused turn as an error carrying the response body.
// A close that names no call is the typed interview's own.
export function turnRefusalOf(error: Error | undefined): HeldBy | null {
  if (!error?.message) return null;
  let body: unknown;
  try {
    body = JSON.parse(error.message);
  } catch {
    return null;
  }
  if (!body || typeof body !== "object") return null;
  const heldBy = heldByOf((body as { detail?: unknown }).detail);
  return heldBy?.callId ? heldBy : null;
}

function heldByOf(detail: unknown): HeldBy | null {
  if (!detail || typeof detail !== "object") return null;
  const { code, call_id } = detail as { code?: unknown; call_id?: unknown };
  const callId = typeof call_id === "string" && call_id ? call_id : null;
  if (code === CALL_IN_PROGRESS) {
    return callId ? { kind: "call_in_progress", callId } : null;
  }
  if (code === INTERVIEW_CLOSED) return { kind: "closed", callId };
  return null;
}

// Someone who chose to type keeps typing when the step opens again in this
// tab: a reload or Back lands on their typed interview, not on the choice.
export function rememberTyping(userId: string | null) {
  if (!userId) return;
  try {
    sessionStorage.setItem(`${TYPING_KEY_PREFIX}${userId}`, "1");
  } catch {
    // The choice is offered again next time.
  }
}

export function isTypingRemembered(userId: string | null) {
  if (!userId || typeof window === "undefined") return false;
  try {
    return sessionStorage.getItem(`${TYPING_KEY_PREFIX}${userId}`) === "1";
  } catch {
    return false;
  }
}

export function forgetTyping(userId: string | null) {
  if (!userId) return;
  try {
    sessionStorage.removeItem(`${TYPING_KEY_PREFIX}${userId}`);
  } catch {
    // Nothing kept, nothing to forget.
  }
}

function retryAfterSecondsOf(error: ApiError) {
  const detail: unknown = error.response?.detail;
  if (!detail || typeof detail !== "object") return null;
  const seconds = (detail as { retry_after_seconds?: unknown })
    .retry_after_seconds;
  return typeof seconds === "number" && seconds > 0 ? seconds : null;
}

function isHiddenMessage(message: UIMessage) {
  const metadata = message.metadata;
  return (
    message.role === "user" &&
    !!metadata &&
    typeof metadata === "object" &&
    (metadata as { hidden?: unknown }).hidden === true
  );
}
