"use client";

import {
  useGetSubscriptionStatus,
  useGetV1ManagePaymentMethods,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { GetV1ManagePaymentMethods200 } from "@/app/api/__generated__/models/getV1ManagePaymentMethods200";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { toast } from "@/components/molecules/Toast/use-toast";

import { formatShortDate } from "../../../helpers";

const PLAN_LABEL: Record<string, string> = {
  NO_TIER: "No active subscription",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
};

// Team upgrade is contact-sales — open this page rather than firing a
// Stripe Checkout from the BUSINESS tier. Replace once marketing publishes
// the canonical sales URL.
export const TEAM_UPGRADE_URL = "https://agpt.co/contact-sales";

// User-visible paid plans only. NO_TIER is the "no active subscription" state
// (gates the platform via PaywallGate); BASIC + ENTERPRISE are reserved
// internal slots and never offered as upgrade targets in the settings UI.
const TIER_ORDER = [
  "PRO",
  "MAX",
  "BUSINESS",
] as const satisfies readonly SubscriptionTierRequestTier[];

function getNextTier(current: string): SubscriptionTierRequestTier | null {
  const idx = TIER_ORDER.indexOf(current as (typeof TIER_ORDER)[number]);
  if (idx === -1) return TIER_ORDER[0];
  if (idx === TIER_ORDER.length - 1) return null;
  return TIER_ORDER[idx + 1];
}

function getPreviousTier(current: string): SubscriptionTierRequestTier | null {
  const idx = TIER_ORDER.indexOf(current as (typeof TIER_ORDER)[number]);
  if (idx <= 0) return null;
  return TIER_ORDER[idx - 1];
}

export function useYourPlanCard() {
  const subscription = useGetSubscriptionStatus({
    query: {
      select: (res) =>
        res.status === 200
          ? (res.data as SubscriptionStatusResponse)
          : undefined,
    },
  });

  const paymentPortal = useGetV1ManagePaymentMethods({
    query: {
      select: (res) => {
        const raw = res.data as GetV1ManagePaymentMethods200 | undefined;
        const url = raw && typeof raw === "object" ? raw.url : undefined;
        return typeof url === "string" ? url : undefined;
      },
    },
  });

  const { mutateAsync: updateTier, isPending: isUpdatingTier } =
    useUpdateSubscriptionTier();

  const effectiveTier = subscription.data?.tier ?? null;
  const isPaid = effectiveTier !== null && effectiveTier !== "NO_TIER";

  const nextTierKey = effectiveTier ? getNextTier(effectiveTier) : null;
  const previousTierKey = effectiveTier ? getPreviousTier(effectiveTier) : null;

  const pendingTier = subscription.data?.pending_tier ?? null;
  const pendingEffectiveAt =
    subscription.data?.pending_tier_effective_at ?? null;
  // Cancellation = pending change to the "no active subscription" state.
  const isPendingCancel = pendingTier === "NO_TIER";
  // Paid→paid downgrade scheduled in Stripe (e.g. MAX → PRO at period end).
  // Initiated from the Stripe billing portal — there's no in-app trigger.
  const isPendingDowngrade = pendingTier !== null && pendingTier !== "NO_TIER";

  const plan = subscription.data
    ? {
        tierKey: effectiveTier ?? subscription.data.tier,
        label:
          PLAN_LABEL[effectiveTier ?? subscription.data.tier] ??
          subscription.data.tier,
        monthlyCostCents: subscription.data.monthly_cost,
        isPaidPlan: isPaid,
        nextTier: nextTierKey,
        nextTierLabel: nextTierKey
          ? (PLAN_LABEL[nextTierKey] ?? nextTierKey)
          : null,
        // Team (BUSINESS) is contact-sales, not a self-serve Checkout. The
        // upgrade button for MAX users opens the marketing/sales page rather
        // than POSTing to /credits/subscription.
        nextTierIsTeamLink: nextTierKey === "BUSINESS",
        previousTier: previousTierKey,
        previousTierLabel: previousTierKey
          ? (PLAN_LABEL[previousTierKey] ?? previousTierKey)
          : null,
        currentPeriodEnd: subscription.data.current_period_end ?? null,
        pendingTier,
        pendingTierLabel: pendingTier
          ? (PLAN_LABEL[pendingTier] ?? pendingTier)
          : null,
        pendingEffectiveAt,
        isPendingCancel,
        isPendingDowngrade,
      }
    : undefined;

  async function changeTier(tier: SubscriptionTierRequestTier) {
    const successUrl = `${window.location.origin}${window.location.pathname}?subscription=success`;
    const cancelUrl = `${window.location.origin}${window.location.pathname}?subscription=cancelled`;
    try {
      const result = await updateTier({
        data: {
          tier,
          success_url: successUrl,
          cancel_url: cancelUrl,
        },
      });
      const url = (result?.data as { url?: string } | undefined)?.url;
      if (url) {
        // Navigating away — don't refetch (would set state on an
        // unmounting component while Stripe Checkout takes over).
        window.location.href = url;
        return;
      }
      await subscription.refetch();
      return true;
    } catch (error) {
      toast({
        title: "Couldn't update your plan",
        description:
          error instanceof Error
            ? error.message
            : "Stripe didn't accept the change. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  }

  async function downgradeSubscription() {
    // End-of-period downgrade: backend's modify_stripe_subscription_for_tier
    // schedules a phase change at current_period_end. No proration today —
    // user keeps higher tier until period end, then drops to lower price.
    if (!plan?.isPaidPlan || !plan?.previousTier) return;
    const targetLabel = plan.previousTierLabel ?? plan.previousTier;
    const ok = await changeTier(plan.previousTier);
    if (!ok) return;
    const periodEnd = plan.currentPeriodEnd
      ? formatShortDate(plan.currentPeriodEnd * 1000)
      : null;
    toast({
      title: "Downgrade scheduled",
      description: periodEnd
        ? `You keep ${plan.label} until ${periodEnd}, then switch to ${targetLabel}. No charge today.`
        : `Your plan will switch to ${targetLabel} at the close of the current billing period. No charge today.`,
    });
  }

  async function resumeSubscription() {
    // POSTing the current tier back to the backend releases any pending
    // schedule — both cancel_at_period_end (NO_TIER pending) and paid→paid
    // downgrade phases. See release_pending_subscription_schedule + the
    // same-tier branch of update_subscription_tier.
    if (
      !plan?.isPaidPlan ||
      (!plan?.isPendingCancel && !plan?.isPendingDowngrade)
    ) {
      return;
    }
    const ok = await changeTier(plan.tierKey as SubscriptionTierRequestTier);
    if (!ok) return;
    toast({
      title: plan.isPendingCancel
        ? "Subscription resumed"
        : "Downgrade cancelled",
      description: `Your ${plan.label} plan will continue to renew as normal.`,
    });
  }

  return {
    plan,
    isLoading: subscription.isLoading,
    isUpdatingTier,
    canManagePortal: Boolean(paymentPortal.data),
    // Don't offer upgrade alongside Resume — pending cancel/downgrade should
    // be released first via resumeSubscription, not stacked with a new tier.
    canUpgrade: Boolean(
      plan?.nextTier && !plan?.isPendingCancel && !plan?.isPendingDowngrade,
    ),
    // Downgrade only when an active paid sub has a tier below it AND no
    // pending change is already in flight — avoids stacking schedules.
    canDowngrade: Boolean(
      plan?.isPaidPlan &&
        plan?.previousTier &&
        !plan?.isPendingCancel &&
        !plan?.isPendingDowngrade,
    ),
    canResume: Boolean(plan?.isPendingCancel || plan?.isPendingDowngrade),
    onUpgrade: () => {
      if (!plan?.nextTier) return;
      // Team (BUSINESS) tier is contact-sales — divert to marketing page
      // instead of POSTing a Checkout the user can't self-serve.
      if (plan.nextTierIsTeamLink) {
        window.open(TEAM_UPGRADE_URL, "_blank", "noopener,noreferrer");
        return;
      }
      void changeTier(plan.nextTier);
    },
    onDowngrade: () => {
      void downgradeSubscription();
    },
    onResume: () => {
      void resumeSubscription();
    },
    onManage: () => {
      if (paymentPortal.data) window.location.href = paymentPortal.data;
    },
  };
}
