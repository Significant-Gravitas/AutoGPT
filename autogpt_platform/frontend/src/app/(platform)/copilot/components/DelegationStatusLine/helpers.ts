import {
  type ChatDelegation,
  type LiveDelegationStatus,
  delegationName,
  formatElapsed,
} from "../../delegations";

export interface StatusLineText {
  /** The bold part: who and what state. */
  headline: string;
  /** The muted tail: timing, the question, the error. */
  detail: string | null;
}

function shortPrompt(prompt: string | null): string | null {
  if (!prompt) return null;
  const line = prompt.split("\n")[0].trim();
  return line.length > 60 ? `${line.slice(0, 57)}…` : line;
}

export function getStatusLineText(
  delegation: ChatDelegation,
  status: LiveDelegationStatus,
  elapsedSeconds: number | null,
  question: string | null,
): StatusLineText {
  const name = delegationName(delegation);
  const elapsed = formatElapsed(elapsedSeconds);
  switch (status) {
    case "proposed":
      return {
        headline: `${name} is waiting for your approval`,
        detail: shortPrompt(delegation.prompt),
      };
    case "queued":
      return {
        headline: `${name} queued`,
        detail: "Starts as soon as they are free",
      };
    case "running":
      return {
        headline: `${name} working`,
        detail: [shortPrompt(delegation.prompt), elapsed]
          .filter(Boolean)
          .join(" · "),
      };
    case "needs-input":
      return { headline: `${name} needs you`, detail: question };
    case "completed": {
      const files = delegation.files.length;
      return {
        headline: `${name} reported back`,
        detail: [
          elapsed,
          files > 0 ? `${files} file${files === 1 ? "" : "s"}` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      };
    }
    case "transferred":
      return { headline: `Handed over to ${name}`, detail: null };
    case "failed":
      return { headline: `${name} stopped`, detail: delegation.error };
    case "cancelled":
      return { headline: `${name} was stopped`, detail: null };
    case "unknown":
      return {
        headline: `${name} · status unclear`,
        detail: "Live updates stopped; open the thread to check.",
      };
  }
}

export function retryMessage(delegation: ChatDelegation): string {
  return `Please retry the hand-off to ${delegationName(delegation)}.`;
}
