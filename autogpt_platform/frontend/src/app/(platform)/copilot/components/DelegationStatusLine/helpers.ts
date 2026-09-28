import type { ChatDelegation, LiveDelegationStatus } from "../../delegations";
import { formatElapsed } from "../../delegations";
import { FALLBACK_EXPERT_NAME, formatCost } from "../../delegationViews";

export interface StatusLineText {
  /** The bold part: who and what state. */
  headline: string;
  /** The muted tail: timing, the question, the error. */
  detail: string | null;
}

export interface StatusLineInput {
  name: string;
  status: LiveDelegationStatus;
  elapsedSeconds: number | null;
  question: string | null;
  /** The user answered the teammate's question from this chat. */
  resumed: boolean;
}

function joined(parts: (string | null | false)[]): string | null {
  const kept = parts.filter((part): part is string => !!part);
  return kept.length > 0 ? kept.join(" · ") : null;
}

export function getStatusLineText(
  delegation: ChatDelegation,
  { name, status, elapsedSeconds, question, resumed }: StatusLineInput,
): StatusLineText {
  const elapsed = formatElapsed(elapsedSeconds);
  const cost = formatCost(delegation.costUsd);
  switch (status) {
    case "proposed":
      return {
        headline: `${name} is waiting for your approval`,
        detail: null,
      };
    case "queued":
      return {
        headline: "1 expert queued",
        detail: `${name} starts as soon as they are free`,
      };
    case "running":
      return {
        headline: "1 expert working",
        detail: joined([name, resumed && "resumed", elapsed, cost]),
      };
    case "needs-input":
      return { headline: `${name} needs you`, detail: question };
    case "completed": {
      const files = delegation.files.length;
      return {
        headline: `${name} reported back`,
        detail: joined([
          elapsed,
          cost,
          files > 0 && `${files} file${files === 1 ? "" : "s"}`,
        ]),
      };
    }
    case "transferred":
      return { headline: `Handed over to ${name}`, detail: null };
    case "failed":
      return {
        headline: `${name} stopped`,
        detail: joined([delegation.error, elapsed, cost]),
      };
    case "cancelled":
      return {
        headline: `You stopped ${name === FALLBACK_EXPERT_NAME ? "your expert" : name}`,
        detail: null,
      };
    case "unknown":
      return {
        headline: `${name} · status unclear`,
        detail: "Live updates stopped; open the thread to check.",
      };
  }
}

export function retryMessage(name: string): string {
  return `Please retry the hand-off to ${name}.`;
}
