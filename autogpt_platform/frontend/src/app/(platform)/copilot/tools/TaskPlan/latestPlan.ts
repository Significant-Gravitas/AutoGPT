"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { createContext } from "react";
import { getProposedPlan } from "./helpers";

/** The id of the newest plan proposed in the session. An older plan card is
 *  replaced by it: greyed out, with no way to approve it. */
export const LatestPlanContext = createContext<string | null>(null);

export function getLatestPlanId(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): string | null {
  let latest: string | null = null;
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const part of message.parts) {
      if (part.type !== "tool-propose_plan" || !("output" in part)) continue;
      const plan = getProposedPlan(part.output);
      if (plan) latest = plan.id;
    }
  }
  return latest;
}
