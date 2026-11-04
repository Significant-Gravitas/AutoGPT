"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { createContext } from "react";

type SentDecision = "approved" | "declined";

const DECISION_LINE =
  /^(Approved|Not approved):.*\(confirmation_id: ([^)\s]+)\)\.?$/gm;

/** The hire/raise answers the user already sent, by confirmation id, so a
 *  proposal card stays answered after a reload. */
export const SentDecisionsContext = createContext<
  ReadonlyMap<string, SentDecision>
>(new Map());

export function getSentDecisions(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  const decisions = new Map<string, SentDecision>();
  for (const message of messages) {
    if (message.role !== "user") continue;
    for (const part of message.parts) {
      if (part.type !== "text") continue;
      for (const match of part.text.matchAll(DECISION_LINE)) {
        decisions.set(
          match[2],
          match[1] === "Approved" ? "approved" : "declined",
        );
      }
    }
  }
  return decisions;
}
