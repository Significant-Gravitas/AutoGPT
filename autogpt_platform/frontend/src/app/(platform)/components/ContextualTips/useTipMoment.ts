"use client";

import { useEffect, useRef } from "react";
import { requestTip } from "./store";
import { mentionsCost, type TipID } from "./tips";

// Reports a tip's moment once `happened` turns true. The host decides when
// (and whether) to show it.
export function useTipMoment(id: TipID, happened: boolean) {
  useEffect(() => {
    if (happened) requestTip(id);
  }, [id, happened]);
}

interface LiveMomentArgs {
  happened: boolean;
  /** The turn that shows it is streaming in front of the user right now. */
  isLive: boolean;
}

// For a moment that also sits in chat history: it counts only once it has
// arrived live in this view, never when an old chat is read back.
export function useLiveTipMoment(
  id: TipID,
  { happened, isLive }: LiveMomentArgs,
) {
  const sawLive = useRef(false);

  useEffect(() => {
    if (isLive) sawLive.current = true;
    if (happened && sawLive.current) requestTip(id);
  }, [id, happened, isLive]);
}

interface MessageLike {
  role: string;
  parts: ReadonlyArray<{ type: string }>;
}

function isTextPart(part: { type: string }): part is {
  type: "text";
  text: string;
} {
  return part.type === "text" && "text" in part;
}

// Otto's latest finished reply talks about what something costs. Only the
// newest reply counts: scrolling back through an old chat is not a moment.
export function latestReplyMentionsCost(
  messages: ReadonlyArray<MessageLike>,
  isStreaming: boolean,
) {
  if (isStreaming) return false;
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return false;
  return last.parts.some((part) => isTextPart(part) && mentionsCost(part.text));
}

interface CostReplyArgs {
  messages: ReadonlyArray<MessageLike>;
  isStreaming: boolean;
  enabled: boolean;
}

// The moment is Otto saying it, so only a reply that finishes streaming in
// front of the user counts. Opening a chat whose history already mentions a
// cost is reading back, not the first time it came up.
export function useCostReplyMoment({
  messages,
  isStreaming,
  enabled,
}: CostReplyArgs) {
  const sawReplyStream = useRef(false);

  useEffect(() => {
    if (isStreaming) {
      sawReplyStream.current = true;
      return;
    }
    if (!sawReplyStream.current) return;
    sawReplyStream.current = false;
    if (enabled && latestReplyMentionsCost(messages, false))
      requestTip("first-cost");
  }, [isStreaming, messages, enabled]);
}
