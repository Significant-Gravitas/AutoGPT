import * as Sentry from "@sentry/nextjs";
import type { ReplyRecord } from "./replyTiming";

/**
 * Report a reply as its own "copilot reply" transaction, from the send to
 * its first answer text (or to where it ended without one), so Explore can
 * set it beside the backend's `copilot.turn` transactions. Best effort: the
 * reply is already on screen, so nothing here may throw into the chat.
 */
export function recordReplySpan(record: ReplyRecord) {
  const { timeOrigin, startedAt } = record.start;
  try {
    Sentry.startInactiveSpan({
      name: "copilot reply",
      op: "copilot.reply",
      forceTransaction: true,
      startTime: timeOrigin + startedAt,
      attributes: replySpanAttributes(record),
    }).end(timeOrigin + record.endedAt);
  } catch {
    // Dropped: a timing sample is never worth a broken chat.
  }
}

function replySpanAttributes(record: ReplyRecord) {
  const attributes: Record<string, string | number | boolean> = {
    "copilot.reply.outcome": record.outcome,
    "copilot.reply.new_chat": record.start.isNewChat,
    "copilot.reply.expert": record.start.isExpert,
    "copilot.reply.stream_client": record.streamClient,
    // Waited behind the user's other running turns: the reply's time
    // includes that wait.
    "copilot.reply.queued": record.wasQueued,
  };
  if (record.firstOutput) {
    attributes["copilot.reply.first_output_ms"] = Math.round(
      record.firstOutput.ms,
    );
    attributes["copilot.reply.first_output_kind"] = record.firstOutput.kind;
  }
  if (record.firstTextMs !== null) {
    attributes["copilot.reply.first_text_ms"] = Math.round(record.firstTextMs);
  }
  return attributes;
}
