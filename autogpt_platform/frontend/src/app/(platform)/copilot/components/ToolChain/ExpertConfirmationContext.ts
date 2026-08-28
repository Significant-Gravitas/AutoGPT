import { createContext } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";

const EMPTY_CONFIRMATIONS = new Set<string>();

export const ExpertConfirmationContext =
  createContext<ReadonlySet<string>>(EMPTY_CONFIRMATIONS);

export function getAppliedExpertConfirmationIDs(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): ReadonlySet<string> {
  const confirmationIDs = new Set<string>();
  for (const message of messages) {
    for (const part of message.parts) {
      if (!("output" in part)) continue;
      const output = objectOutput(part.output);
      if (!output) continue;
      addSingleConfirmation(output, confirmationIDs);
      addBatchConfirmations(output, confirmationIDs);
    }
  }
  return confirmationIDs;
}

function addSingleConfirmation(
  output: Record<string, unknown>,
  confirmationIDs: Set<string>,
) {
  if (output.type !== "expert_change_applied" || output.applied !== true)
    return;
  const confirmationID = stringValue(output.confirmation_id);
  if (confirmationID) confirmationIDs.add(confirmationID);
}

function addBatchConfirmations(
  output: Record<string, unknown>,
  confirmationIDs: Set<string>,
) {
  if (!Array.isArray(output.results)) return;
  for (const value of output.results) {
    const result = objectOutput(value);
    if (
      !result ||
      !["applied", "already_applied"].includes(String(result.outcome))
    ) {
      continue;
    }
    const confirmationID = stringValue(result.confirmation_id);
    if (confirmationID) confirmationIDs.add(confirmationID);
  }
}

function objectOutput(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      return objectOutput(JSON.parse(value));
    } catch {
      return null;
    }
  }
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
