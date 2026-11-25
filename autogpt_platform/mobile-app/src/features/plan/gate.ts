// No native imports, so tests can load this without a phone.
import type { PlanTier } from "../../lib/api/billing";

export type PlanGate = "loading" | "blocked" | "open";

export interface PlanCheck {
  /** False on a server that sells no plans (local, self-hosted). */
  isPaymentEnabled: boolean;
  hasPlan: boolean;
}

/** How long each question to the server may take before the person goes in. */
export const PLAN_CHECK_TIMEOUT_MS = 8_000;

const ACTIVATION_RETRIES = 3;

/**
 * The web's PaywallGate rule: the stored tier, not whether Stripe answers
 * right now, so a Stripe outage never locks out someone who pays. Trials
 * and hand-set plans count.
 */
export function readPlanCheck(
  isPaymentEnabled: boolean,
  tier: PlanTier | null,
): PlanCheck {
  if (!isPaymentEnabled) return { isPaymentEnabled, hasPlan: true };
  return { isPaymentEnabled, hasPlan: tier !== "NO_TIER" };
}

/**
 * Where a signed-in account goes. Once answered, the answer stands through a
 * failed re-check. A server that never answers lets the person in: it still
 * refuses work (402) to an account with no plan.
 */
export function planGate({
  check,
  status,
}: {
  check: PlanCheck | undefined;
  status: "pending" | "error" | "success";
}): PlanGate {
  if (check) return check.hasPlan ? "open" : "blocked";
  return status === "pending" ? "loading" : "open";
}

/** Asks again only while a new plan is still being set up (503). */
export function shouldRetryPlanCheck(failureCount: number, error: unknown) {
  const status = (error as { status?: unknown } | null)?.status;
  return status === 503 && failureCount < ACTIVATION_RETRIES;
}
