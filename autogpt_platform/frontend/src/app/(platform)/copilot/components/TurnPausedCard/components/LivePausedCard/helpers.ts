import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";

import {
  parseProviderFailure,
  type ProviderFailure,
} from "../../../../providerFailure";
import type { TurnPause } from "../../../../turnPause";
import { neverResets } from "../../helpers";

/** A budget pause waits on our own allowance, so the platform is the
 *  connection that's out and never the one to continue on. */
export const PLATFORM_BUDGET: ProviderFailure = {
  kind: "usage_limit",
  message: "Usage limit reached",
  authProvider: "platform",
  credentialId: null,
  resetsAt: null,
  retryable: false,
  reconnectFixesIt: false,
};

/** A task can only be picked back up where tools run: ChatGPT runs them,
 *  Microsoft 365 Copilot returns text only. */
export function canContinueTask(offer: AIConnectionOffer): boolean {
  return offer.auth_provider === "codex";
}

interface PrimaryArgs {
  pause: TurnPause;
  budgetBack: boolean;
  resume: { resumesAt: Date | null };
}

/** What the card leads with. A trial, or a budget that never refills, has
 *  only Upgrade (and ChatGPT) as ways on; a resume already set shows in place
 *  of the button that would set it. */
export function primaryAction({
  pause,
  budgetBack,
  resume,
}: PrimaryArgs): "keep_going" | "resume" | null {
  if (pause.reason === "trial" || neverResets(pause)) return null;
  if (pause.reason === "task" || pause.reason === "steps" || budgetBack) {
    return "keep_going";
  }
  return resume.resumesAt ? null : "resume";
}

/** The same task, now waiting on the budget the account ran out of after
 *  it paused, when "Keep going" was refused for it. */
export function pauseAfterRefusal(
  pause: TurnPause,
  window: string | null,
  resetsAt: string | null,
): TurnPause {
  if (window === "trial") {
    return { ...pause, reason: "trial", resets_at: null, steps: null };
  }
  return {
    ...pause,
    reason: window === "weekly" ? "weekly" : "daily",
    resets_at: resetsAt,
    steps: null,
  };
}

export type ContinueOutcome =
  | { kind: "running" }
  | { kind: "out_of_budget"; resetsAt: string | null }
  | { kind: "failed"; description: string };

export function continueOutcome(error: unknown): ContinueOutcome {
  const failure = apiFailureOf(error);
  if (failure?.status === 409 && failure.detail === "unsupported_connection") {
    return {
      kind: "failed",
      description:
        "This chat now runs on Microsoft 365 Copilot, which can't pick up a paused task. Switch its connection back to keep going.",
    };
  }
  // One turn per chat: the one already running is the one to follow.
  if (failure?.status === 409) return { kind: "running" };
  const limit =
    failure?.status === 429 ? parseProviderFailure(failure.detail) : null;
  if (limit?.kind === "usage_limit") {
    return {
      kind: "out_of_budget",
      resetsAt:
        limit.resetsAt !== null
          ? new Date(limit.resetsAt * 1000).toISOString()
          : null,
    };
  }
  if (failure?.status === 402) {
    return { kind: "failed", description: "Choose a plan to keep going." };
  }
  // The running-turn cap explains itself.
  if (failure?.status === 429 && typeof failure.detail === "string") {
    return { kind: "failed", description: failure.detail };
  }
  return { kind: "failed", description: "Try again in a moment." };
}

export type AtResetOutcome =
  | { kind: "budget_back" }
  | { kind: "failed"; title: string; description: string };

export function atResetOutcome(error: unknown): AtResetOutcome {
  const failure = apiFailureOf(error);
  if (failure?.status === 409 && failure.detail === "not_out_of_budget") {
    return { kind: "budget_back" };
  }
  if (failure?.status === 409 && failure.detail === "no_reset") {
    return {
      kind: "failed",
      title: "This allowance doesn't reset",
      description: "Upgrade to keep going.",
    };
  }
  return {
    kind: "failed",
    title:
      failure?.status === 503
        ? "Couldn't check your budget"
        : "Couldn't set the resume",
    description: "Try again in a moment.",
  };
}

interface ApiFailure {
  status: number;
  detail: unknown;
}

/** The status and FastAPI `detail` of a call the generated client rejected. */
function apiFailureOf(error: unknown): ApiFailure | null {
  if (!error || typeof error !== "object") return null;
  const { status, response } = error as {
    status?: unknown;
    response?: unknown;
  };
  if (typeof status !== "number") return null;
  const detail =
    response && typeof response === "object"
      ? (response as { detail?: unknown }).detail
      : undefined;
  return { status, detail };
}
