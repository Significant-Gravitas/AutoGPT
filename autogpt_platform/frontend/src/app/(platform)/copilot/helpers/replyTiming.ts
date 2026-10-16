import type { ChatStatus, UIMessage } from "ai";
import { getVisiblePartKind, type VisiblePartKind } from "../helpers";
import { parseSpecialMarkers } from "./messageMarkers";

export type StreamClient = "ai-sdk" | "stream-runtime";
export type ReplyOutcome = "answered" | "no_text" | "abandoned";

/** The user's send, carried to whichever chat host dispatches it. */
export interface ReplyTimingStart {
  /** `performance.now()` at the send. */
  startedAt: number;
  /** The clock `startedAt` was read on; a full navigation starts a new one. */
  timeOrigin: number;
  isNewChat: boolean;
  isExpert: boolean;
}

interface FirstOutput {
  ms: number;
  kind: VisiblePartKind;
}

export interface ReplyTracking {
  start: ReplyTimingStart;
  sessionId: string | null;
  streamClient: StreamClient;
  /** What was on screen when the send went out; the reply comes after it. */
  priorIds: ReadonlySet<string>;
  sentId: string | null;
  wasActive: boolean;
  hasReturned: boolean;
  firstOutput: FirstOutput | null;
  /** The turn was seen over without text: when, and how it would end. */
  ending: ReplyEnding | null;
}

interface ReplyEnding {
  at: number;
  outcome: ReplyOutcome;
}

export interface ReplyView {
  sessionId: string | null;
  messages: UIMessage[];
  status: ChatStatus;
  /** Nothing in flight, probing or reconnecting: the turn is over on screen. */
  isSettled: boolean;
  isUserStopping: boolean;
}

export interface ReplyRecord {
  start: ReplyTimingStart;
  streamClient: StreamClient;
  endedAt: number;
  outcome: ReplyOutcome;
  firstOutput: FirstOutput | null;
  firstTextMs: number | null;
}

/** A null `tracking` ends the timing; a null `record` ends it unreported. */
export interface ReplyStep {
  tracking: ReplyTracking | null;
  record: ReplyRecord | null;
  /** Check again after this long even if nothing renders meanwhile. */
  recheckInMs?: number;
}

// The screen can trail a turn that has ended: the stream runtime reveals the
// last of its text a tick after the turn finishes.
export const SETTLE_GRACE_MS = 250;

const DROPPED: ReplyStep = { tracking: null, record: null };

export function startReplyTiming(flags: {
  isNewChat: boolean;
  isExpert: boolean;
}): ReplyTimingStart {
  return {
    startedAt: performance.now(),
    timeOrigin: performance.timeOrigin,
    ...flags,
  };
}

/** Null for a send from before a full navigation: its clock has restarted. */
export function beginReplyTracking(
  start: ReplyTimingStart,
  view: ReplyView,
  streamClient: StreamClient,
): ReplyTracking | null {
  if (start.timeOrigin !== performance.timeOrigin) return null;
  return {
    start,
    sessionId: view.sessionId,
    streamClient,
    priorIds: new Set(view.messages.map((message) => message.id)),
    sentId: null,
    wasActive: false,
    hasReturned: false,
    firstOutput: null,
    ending: null,
  };
}

export function advanceReply(
  tracking: ReplyTracking,
  view: ReplyView,
  now: number,
): ReplyStep {
  if (view.isUserStopping || view.sessionId !== tracking.sessionId) {
    return abandonReply(tracking, now);
  }
  const sentIndex = findSentMessage(view.messages, tracking);
  if (sentIndex === -1 && tracking.sentId) {
    // Once its turn is over, the list is rebuilt from the server under new
    // ids. Before that, the send was taken back: refused (a 429 or 402 puts
    // it back in the composer), or queued behind a turn this tab missed.
    const { ending } = tracking;
    return ending ? finish(tracking, ending.at, ending.outcome, null) : DROPPED;
  }

  const reply = scanReply(
    sentIndex === -1 ? [] : view.messages.slice(sentIndex + 1),
  );
  const elapsed = now - tracking.start.startedAt;
  const next: ReplyTracking = {
    ...tracking,
    sentId: sentIndex === -1 ? null : view.messages[sentIndex].id,
    wasActive:
      tracking.wasActive ||
      view.status === "submitted" ||
      view.status === "streaming",
    firstOutput:
      tracking.firstOutput ??
      (reply.firstKind ? { ms: elapsed, kind: reply.firstKind } : null),
  };
  if (reply.hasText) return finish(next, now, "answered", elapsed);

  const isOver = view.isSettled && (next.wasActive || next.hasReturned);
  if (!isOver) return { tracking: { ...next, ending: null }, record: null };
  const failed = view.status === "error" || reply.hasErrorMarker;
  const ending: ReplyEnding = next.ending ?? {
    at: now,
    outcome: failed ? "abandoned" : "no_text",
  };
  const waited = now - ending.at;
  if (waited < SETTLE_GRACE_MS) {
    return {
      tracking: { ...next, ending },
      record: null,
      recheckInMs: SETTLE_GRACE_MS - waited,
    };
  }
  // A send that never showed was suppressed (reconnecting, duplicate).
  if (!next.sentId) return DROPPED;
  return finish(next, ending.at, ending.outcome, null);
}

/** Given up before its text: reported once its message showed, else dropped. */
export function abandonReply(tracking: ReplyTracking, now: number): ReplyStep {
  return tracking.sentId ? finish(tracking, now, "abandoned", null) : DROPPED;
}

function finish(
  tracking: ReplyTracking,
  endedAt: number,
  outcome: ReplyOutcome,
  firstTextMs: number | null,
): ReplyStep {
  const { start, streamClient, firstOutput } = tracking;
  return {
    tracking: null,
    record: { start, streamClient, endedAt, outcome, firstOutput, firstTextMs },
  };
}

function findSentMessage(messages: UIMessage[], tracking: ReplyTracking) {
  if (tracking.sentId) {
    return messages.findIndex((message) => message.id === tracking.sentId);
  }
  const afterPrior =
    messages.findLastIndex((message) => tracking.priorIds.has(message.id)) + 1;
  return messages.findIndex(
    (message, index) => index >= afterPrior && message.role === "user",
  );
}

/** What the assistant has shown after the sent message, in on-screen order. */
function scanReply(reply: UIMessage[]) {
  const scan = {
    firstKind: null as VisiblePartKind | null,
    hasText: false,
    hasErrorMarker: false,
  };
  for (const message of reply) {
    if (message.role !== "assistant") continue;
    for (const part of message.parts) {
      let kind = getVisiblePartKind(part);
      if (kind === "text" && part.type === "text") {
        const { markerType, cleanText } = parseSpecialMarkers(part.text);
        scan.hasErrorMarker ||=
          markerType === "error" || markerType === "retryable_error";
        // A marker renders as an error card or a system note, not an answer.
        if (!cleanText.trim()) kind = null;
      }
      scan.firstKind ??= kind;
      scan.hasText ||= kind === "text";
    }
  }
  return scan;
}
