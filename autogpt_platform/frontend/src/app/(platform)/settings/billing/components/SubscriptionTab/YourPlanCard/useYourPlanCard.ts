"use client";

import { useEffect, useState } from "react";

import {
  useGetSubscriptionStatus,
  useGetV1ManagePaymentMethods,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { GetV1ManagePaymentMethods200 } from "@/app/api/__generated__/models/getV1ManagePaymentMethods200";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequestBillingCycle } from "@/app/api/__generated__/models/subscriptionTierRequestBillingCycle";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";

import { formatCents, formatShortDate } from "../../../helpers";

const PLAN_LABEL: Record<string, string> = {
  NO_TIER: "No active subscription",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
};

// Team upgrade is contact-sales — open the Tally intake form rather than
// firing a Stripe Checkout from the BUSINESS tier.
export const TEAM_UPGRADE_URL = "https://tally.so/r/2Eb9zj";

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

  const serverCycle: SubscriptionTierRequestBillingCycle =
    subscription.data?.billing_cycle === "yearly" ? "yearly" : "monthly";
  const [selectedCycle, setSelectedCycle] =
    useState<SubscriptionTierRequestBillingCycle>(serverCycle);
  const [pendingCycle, setPendingCycle] =
    useState<SubscriptionTierRequestBillingCycle | null>(null);

  // Re-sync local toggle when the server response updates (e.g. after refetch
  // post-mutation). Avoids stale "yearly" pill after a successful switch.
  useEffect(() => {
    setSelectedCycle(serverCycle);
  }, [serverCycle]);

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

  const tierCostsYearly = subscription.data?.tier_costs_yearly ?? {};
  const tierCosts = subscription.data?.tier_costs ?? {};
  const currentTierKey = effectiveTier ?? "NO_TIER";
  const currentYearlyCents = tierCostsYearly[currentTierKey];
  const currentMonthlyCents = tierCosts[currentTierKey];

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
        billingCycle: serverCycle,
      }
    : undefined;

  // ENTERPRISE seats are admin-managed; BASIC is a reserved internal slot
  // (no Stripe sub, no upgrade target) — neither has a user-manageable cycle.
  const isCycleToggleVisible =
    effectiveTier !== "ENTERPRISE" && effectiveTier !== "BASIC";

  async function changeTier(
    tier: SubscriptionTierRequestTier,
    billingCycle?: SubscriptionTierRequestBillingCycle,
  ) {
    const successUrl = `${window.location.origin}${window.location.pathname}?subscription=success`;
    const cancelUrl = `${window.location.origin}${window.location.pathname}?subscription=cancelled`;
    try {
      const result = await updateTier({
        data: {
          tier,
          success_url: successUrl,
          cancel_url: cancelUrl,
          ...(billingCycle ? { billing_cycle: billingCycle } : {}),
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
      // 422 fail-closed = LD missing the *_YEARLY price for this tier. Surface
      // a human-readable toast and leave the toggle pinned to its prior cycle.
      if (error instanceof ApiError && error.status === 422 && billingCycle) {
        toast({
          title: "Yearly billing is not yet available for your plan.",
          description: "Please contact support.",
          variant: "destructive",
        });
        return false;
      }
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
    // Preserve the user's current cycle on downgrade — without forwarding
    // serverCycle, the backend would default the missing billing_cycle to
    // "monthly" and silently flip a yearly subscriber to monthly at period
    // end (Sentry caught this; the same-tier release path also bites).
    const ok = await changeTier(plan.previousTier, serverCycle);
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
    // Same-tier release also gates on cycle: omitting billing_cycle defaults
    // the backend to "monthly", which on a yearly sub is treated as a
    // cycle-switch instead of a release-pending — silent flip to monthly.
    const ok = await changeTier(
      plan.tierKey as SubscriptionTierRequestTier,
      serverCycle,
    );
    if (!ok) return;
    toast({
      title: plan.isPendingCancel
        ? "Subscription resumed"
        : "Downgrade cancelled",
      description: `Your ${plan.label} plan will continue to renew as normal.`,
    });
  }

  function onCycleChange(nextCycle: SubscriptionTierRequestBillingCycle) {
    if (nextCycle === selectedCycle) return;
    if (!plan?.isPaidPlan) {
      // NO_TIER — the upgrade flow forwards `selectedCycle` to changeTier.
      setSelectedCycle(nextCycle);
      return;
    }
    if (nextCycle === serverCycle) {
      // User toggled back to their current server cycle while a switch
      // dialog was open elsewhere — just clear the staged change.
      setSelectedCycle(nextCycle);
      setPendingCycle(null);
      return;
    }
    setPendingCycle(nextCycle);
  }

  async function confirmCycleSwitch() {
    if (!pendingCycle || !plan?.tierKey) return;
    const ok = await changeTier(
      plan.tierKey as SubscriptionTierRequestTier,
      pendingCycle,
    );
    setPendingCycle(null);
    if (!ok) return;
    setSelectedCycle(pendingCycle);
    toast({
      title:
        pendingCycle === "yearly"
          ? "Switched to yearly billing"
          : "Switching to monthly at period end",
      description:
        pendingCycle === "yearly"
          ? "Stripe will charge the prorated difference today."
          : "Your plan will switch to monthly at the end of the current period.",
    });
  }

  function cancelCycleSwitch() {
    setPendingCycle(null);
  }

  function getDialogBody(): string {
    if (!pendingCycle) return "";
    if (pendingCycle === "yearly") {
      const yearly =
        currentYearlyCents !== undefined ? formatCents(currentYearlyCents) : "";
      return [
        "You'll be charged the prorated difference immediately.",
        yearly
          ? `After this period, your plan renews yearly at ${yearly}.`
          : "After this period, your plan renews on the new yearly cadence.",
      ].join(" ");
    }
    const monthly =
      currentMonthlyCents !== undefined ? formatCents(currentMonthlyCents) : "";
    return [
      "Your plan will switch to monthly billing at the end of your current yearly period; no charge today.",
      monthly ? `New monthly price: ${monthly}.` : "",
    ]
      .filter(Boolean)
      .join(" ");
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
    selectedCycle,
    pendingCycle,
    isCycleToggleVisible,
    cycleDialogBody: getDialogBody(),
    onCycleChange,
    onConfirmCycleSwitch: () => {
      void confirmCycleSwitch();
    },
    onCancelCycleSwitch: cancelCycleSwitch,
    onUpgrade: () => {
      if (!plan?.nextTier) return;
      // Team (BUSINESS) tier is contact-sales — divert to marketing page
      // instead of POSTing a Checkout the user can't self-serve.
      if (plan.nextTierIsTeamLink) {
        window.open(TEAM_UPGRADE_URL, "_blank", "noopener,noreferrer");
        return;
      }
      void changeTier(plan.nextTier, selectedCycle);
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
