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
