import type { ChatStatus, UIMessage } from "ai";
import {
  isChainableToolPart,
  type MessagePart,
} from "../components/ChatMessagesContainer/helpers";
import {
  isMidTurnFallbackRow,
  splitMessagesAtDrainHints,
} from "../components/ChatMessagesContainer/midTurnSplit";
import { buildChainSegments } from "../components/ToolChain/helpers";
import { getVisiblePartKind, type VisiblePartKind } from "../helpers";
import { isBookkeepingPart } from "../messageParts";
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
  /** The sent message's text, which finds it again under a new id. */
  sentText: string | null;
  wasActive: boolean;
  hasReturned: boolean;
  /** The chat waited behind the user's other running turns. */
  wasQueued: boolean;
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
  /** The chat waits behind the user's other running turns; its turn has not
   *  begun, whatever the stream says. */
  isQueued: boolean;
  isUserStopping: boolean;
}

export interface ReplyRecord {
  start: ReplyTimingStart;
  streamClient: StreamClient;
  endedAt: number;
  outcome: ReplyOutcome;
  wasQueued: boolean;
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

// A turn that showed nothing may only have been queued behind the user's
// other running turns, which the chat's status says once it is read again.
export const QUIET_TURN_GRACE_MS = 3000;

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
    sentText: null,
    wasActive: false,
    hasReturned: false,
    wasQueued: false,
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
  // The rows the page draws: a follow-up drained mid-turn splits its turn.
  const rows = splitMessagesAtDrainHints(view.messages);
  const sentIndex = findSentMessage(rows, tracking);
  if (sentIndex === -1 && tracking.sentId) {
    // The send was taken back: refused (a 429 or 402 puts it back in the
    // composer), or queued behind a turn this tab missed.
    const { ending } = tracking;
    return ending ? finish(tracking, ending.at, ending.outcome, null) : DROPPED;
  }

  const sent = sentIndex === -1 ? null : rows[sentIndex];
  const reply = scanReply(sent ? rows.slice(sentIndex + 1) : []);
  const elapsed = now - tracking.start.startedAt;
  const next: ReplyTracking = {
    ...tracking,
    sentId: sent?.id ?? null,
    sentText: sent ? messageText(sent) : null,
    wasActive:
      tracking.wasActive ||
      view.status === "submitted" ||
      view.status === "streaming",
    wasQueued: tracking.wasQueued || view.isQueued,
    firstOutput:
      tracking.firstOutput ??
      (reply.firstKind ? { ms: elapsed, kind: reply.firstKind } : null),
  };
  if (reply.hasText) return finish(next, now, "answered", elapsed);
  // Another message the user sent showed before any answer did.
  if (reply.isFollowedUp) return abandonReply(next, now);

  const isOver =
    view.isSettled && !view.isQueued && (next.wasActive || next.hasReturned);
  if (!isOver) return { tracking: { ...next, ending: null }, record: null };
  const failed = view.status === "error" || reply.hasErrorMarker;
  const ending: ReplyEnding = next.ending ?? {
    at: now,
    outcome: failed ? "abandoned" : "no_text",
  };
  const grace = next.firstOutput ? SETTLE_GRACE_MS : QUIET_TURN_GRACE_MS;
  const waited = now - ending.at;
  if (waited < grace) {
    return {
      tracking: { ...next, ending },
      record: null,
      recheckInMs: grace - waited,
    };
  }
  // A send that never showed was suppressed (reconnecting, duplicate).
  if (!next.sentId) return DROPPED;
  return finish(next, ending.at, ending.outcome, null);
}

/** Given up before its text: reported once its message showed, else dropped.
 *  A turn already seen over keeps the ending it had. */
export function abandonReply(tracking: ReplyTracking, now: number): ReplyStep {
  if (!tracking.sentId) return DROPPED;
  const { ending } = tracking;
  return ending
    ? finish(tracking, ending.at, ending.outcome, null)
    : finish(tracking, now, "abandoned", null);
}

function finish(
  tracking: ReplyTracking,
  endedAt: number,
  outcome: ReplyOutcome,
  firstTextMs: number | null,
): ReplyStep {
  const { start, streamClient, wasQueued, firstOutput } = tracking;
  return {
    tracking: null,
    record: {
      start,
      streamClient,
      endedAt,
      outcome,
      wasQueued,
      firstOutput,
      firstTextMs,
    },
  };
}

function findSentMessage(rows: UIMessage[], tracking: ReplyTracking) {
  if (tracking.sentId) {
    const index = rows.findIndex((row) => row.id === tracking.sentId);
    if (index !== -1) return index;
    // Rebuilt from the server once its turn is over: the saved copy of the
    // message has a new id.
    return rows.findLastIndex(
      (row) =>
        isOwnUserRow(row, tracking) && messageText(row) === tracking.sentText,
    );
  }
  const afterPrior =
    rows.findLastIndex((row) => tracking.priorIds.has(row.id)) + 1;
  return rows.findIndex(
    (row, index) => index >= afterPrior && isOwnUserRow(row, tracking),
  );
}

function isOwnUserRow(row: UIMessage, tracking: ReplyTracking) {
  return (
    row.role === "user" &&
    !isMidTurnFallbackRow(row) &&
    !tracking.priorIds.has(row.id)
  );
}

/** What the reply showed after the sent message, in on-screen order, up to
 *  the next message the user sent. */
function scanReply(rows: UIMessage[]) {
  const scan = {
    firstKind: null as VisiblePartKind | null,
    hasText: false,
    hasErrorMarker: false,
    isFollowedUp: false,
  };
  for (const row of rows) {
    if (row.role === "user") {
      // A promoted follow-up sits above the live reply it was drained into,
      // so it does not mark where that reply ends.
      if (isMidTurnFallbackRow(row)) continue;
      scan.isFollowedUp = true;
      break;
    }
    if (row.role !== "assistant") continue;
    const parts = row.parts.filter((part) => !isBookkeepingPart(part));
    const folded = foldedIntoChains(parts);
    for (const part of parts) {
      let kind = getVisiblePartKind(part);
      if (kind === "text" && part.type === "text") {
        const { markerType, cleanText } = parseSpecialMarkers(part.text);
        scan.hasErrorMarker ||=
          markerType === "error" || markerType === "retryable_error";
        // A marker renders as an error card or a system note, and narration
        // between tool calls as a row of their chain: neither is the answer.
        if (!cleanText.trim() || folded.has(part)) kind = null;
      }
      scan.firstKind ??= kind;
      scan.hasText ||= kind === "text";
    }
  }
  return scan;
}

function foldedIntoChains(parts: MessagePart[]): ReadonlySet<MessagePart> {
  return new Set(
    buildChainSegments(parts, isChainableToolPart).flatMap((segment) =>
      segment.kind === "chain" ? segment.parts : [],
    ),
  );
}

function messageText(message: UIMessage) {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}
