import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import type { DeleteIntegrationTarget } from "@/components/contextual/IntegrationsPanel/components/hooks/useDeleteIntegration";
import { serviceName } from "@/components/contextual/IntegrationsPanel/helpers";
import { getSubscriptionPlan } from "@/services/billing/subscription-plan";
import { getPlanState, type PendingChange, type PlanState } from "./helpers";

type SubscriptionPlan = ReturnType<typeof getSubscriptionPlan>;

interface PlanContextArgs {
  subscription: SubscriptionStatusResponse | undefined;
  trial: TrialStatusResponse | undefined;
}

interface PendingChangeArgs extends PlanContextArgs {
  planState: PlanState | null;
  plan: SubscriptionPlan;
}

export function uniqueNames(names: string[]): string[] {
  return Array.from(new Set(names));
}

export function getAppNames(credentials: CredentialsMetaResponse[]): string[] {
  return uniqueNames(credentials.map((credential) => serviceName(credential)));
}

export function toDisconnectTargets(
  credentials: CredentialsMetaResponse[],
): DeleteIntegrationTarget[] {
  return credentials.map((credential) => ({
    id: credential.id,
    provider: credential.provider,
    name: serviceName(credential),
  }));
}

export function getPlanContext({ subscription, trial }: PlanContextArgs) {
  const plan = getSubscriptionPlan(subscription);
  const state = getPlanState({
    tier: plan.tier,
    isAdminManaged: plan.isAdminManaged,
  });
  // A trial's pending cancel lives in the trial status, so without it the
  // trial panels can't be honest and none is shown.
  const planState = state === "trial" && !trial ? null : state;
  return {
    tier: plan.tier,
    planState,
    pending: getPendingChange({ planState, plan, subscription, trial }),
    proMonthlyCents: getProMonthlyCents(subscription),
    trialEndsAt: planState === "trial" ? (trial?.ends_at ?? null) : null,
  };
}

function getPendingChange({
  planState,
  plan,
  subscription,
  trial,
}: PendingChangeArgs): PendingChange | null {
  if (planState === "trial") {
    return trial?.cancel_at_period_end && trial.ends_at
      ? { kind: "cancel", effectiveAt: trial.ends_at }
      : null;
  }
  if (plan.isPendingCancel) {
    const effectiveAt = plan.pendingEffectiveAt ?? getPeriodEnd(subscription);
    return effectiveAt ? { kind: "cancel", effectiveAt } : null;
  }
  if (
    planState === "max" &&
    plan.isPendingDowngrade &&
    plan.pendingTier === "PRO" &&
    plan.pendingEffectiveAt
  ) {
    return { kind: "downgrade", effectiveAt: plan.pendingEffectiveAt };
  }
  return null;
}

function getPeriodEnd(subscription: SubscriptionStatusResponse | undefined) {
  const seconds = subscription?.current_period_end;
  return seconds ? seconds * 1000 : null;
}

function getProMonthlyCents(
  subscription: SubscriptionStatusResponse | undefined,
): number | null {
  const cents = subscription?.tier_costs?.PRO;
  return typeof cents === "number" && cents > 0 ? cents : null;
}
