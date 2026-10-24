"use client";

import { useEffect } from "react";
import { requestTip } from "./store";
import { mentionsCost, type TipID } from "./tips";

// Reports a tip's moment once `happened` turns true. The host decides when
// (and whether) to show it.
export function useTipMoment(id: TipID, happened: boolean) {
  useEffect(() => {
    if (happened) requestTip(id);
  }, [id, happened]);
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
