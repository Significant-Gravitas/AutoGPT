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
  // Tier upgrades on a paid plan run through always_invoice immediate billing
  // — without a confirmation step the user is silently charged a large
  // prorated amount on a single click. Stage the target tier here and only
  // fire the mutation after the SwitchTierDialog confirms.
  const [pendingTierUpgrade, setPendingTierUpgrade] =
    useState<SubscriptionTierRequestTier | null>(null);
  // Downgrade is end-of-period (no charge today) so the risk is lower than
  // upgrade, but we still gate behind a confirm dialog so the user gets
  // explicit feedback that the action took effect.
  const [pendingTierDowngrade, setPendingTierDowngrade] =
    useState<SubscriptionTierRequestTier | null>(null);

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
  const pendingBillingCycle = subscription.data?.pending_billing_cycle ?? null;
  // Cancellation = pending change to the "no active subscription" state.
  const isPendingCancel = pendingTier === "NO_TIER";
  // Same-tier yearly→monthly schedule: pending_tier matches current tier but
  // pending_billing_cycle differs from serverCycle. Treat as a cycle-only
  // switch so the badge + copy describe the change accurately rather than
  // claiming a confusing "downgrade to <same tier>".
  const isPendingCycleSwitch =
    pendingTier !== null &&
    pendingTier !== "NO_TIER" &&
    pendingTier === effectiveTier &&
    pendingBillingCycle !== null &&
    pendingBillingCycle !== serverCycle;
  // Paid→paid tier downgrade scheduled in Stripe (e.g. MAX → PRO at period
  // end). The in-app trigger lives on the Downgrade button below; the Stripe
  // billing portal can also originate one.
  const isPendingDowngrade =
    pendingTier !== null &&
    pendingTier !== "NO_TIER" &&
    pendingTier !== effectiveTier;

  const tierCostsYearly = subscription.data?.tier_costs_yearly ?? {};
  const tierCosts = subscription.data?.tier_costs ?? {};
  const currentTierKey = effectiveTier ?? "NO_TIER";
  const currentYearlyCents = tierCostsYearly[currentTierKey];
  const currentMonthlyCents = tierCosts[currentTierKey];
  // Stripe credits the unused portion of the user's current monthly invoice
  // when switching to yearly with proration_behavior="always_invoice"; the
  // backend computes the same value (monthly_cost * remaining / total) and
  // exposes it on the status payload so we can show the actual charge today.
  const prorationCreditCents = subscription.data?.proration_credit_cents ?? 0;
  const currentPeriodEndMs = subscription.data?.current_period_end
    ? subscription.data.current_period_end * 1000
    : null;

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
        pendingCycle: pendingBillingCycle,
        isPendingCancel,
        isPendingDowngrade,
        isPendingCycleSwitch,
        billingCycle: serverCycle,
      }
    : undefined;

  // ENTERPRISE seats are admin-managed; BASIC is a reserved internal slot;
  // NO_TIER means no active subscription. None of those have a
  // user-manageable cycle to switch — the toggle would be dead UI.
  const isCycleToggleVisible =
    effectiveTier !== null &&
    effectiveTier !== "ENTERPRISE" &&
    effectiveTier !== "BASIC" &&
    effectiveTier !== "NO_TIER";

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
      // 422 fail-closed on a YEARLY request = LD missing the *_YEARLY price for
      // this tier. Match billingCycle === "yearly" specifically — a 422 on a
      // monthly request is a different failure (e.g. tier unconfigured) and
      // should surface the generic "Couldn't update your plan" copy below
      // instead of misleading the user about yearly availability.
      if (
        error instanceof ApiError &&
        error.status === 422 &&
        billingCycle === "yearly"
      ) {
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

  async function downgradeSubscription(
    targetTier: SubscriptionTierRequestTier,
  ) {
    // End-of-period downgrade: backend's modify_stripe_subscription_for_tier
    // schedules a phase change at current_period_end. No proration today —
    // user keeps higher tier until period end, then drops to lower price.
    if (!plan?.isPaidPlan) return;
    const targetLabel = PLAN_LABEL[targetTier] ?? targetTier;
    // Preserve the user's current cycle on downgrade — without forwarding
    // serverCycle, the backend would default the missing billing_cycle to
    // "monthly" and silently flip a yearly subscriber to monthly at period
    // end (Sentry caught this; the same-tier release path also bites).
    const ok = await changeTier(targetTier, serverCycle);
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
    // POSTing the current tier+cycle back to the backend releases any pending
    // schedule — cancel_at_period_end (NO_TIER pending), paid→paid downgrade,
    // and same-tier yearly→monthly cycle switch all share the release path.
    // See release_pending_subscription_schedule + the same-tier branch of
    // update_subscription_tier.
    if (
      !plan?.isPaidPlan ||
      (!plan?.isPendingCancel &&
        !plan?.isPendingDowngrade &&
        !plan?.isPendingCycleSwitch)
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
        : plan.isPendingCycleSwitch
          ? "Cycle switch cancelled"
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

  function getDialogTitle(): string {
    if (!pendingCycle) return "";
    const tierLabel = effectiveTier ? PLAN_LABEL[effectiveTier] : null;
    // NO_TIER / unknown tiers don't have a meaningful label for the title
    // ("Switch No active subscription to yearly billing?" reads as nonsense).
    if (
      !tierLabel ||
      effectiveTier === "NO_TIER" ||
      effectiveTier === "BUSINESS"
    ) {
      return pendingCycle === "yearly"
        ? "Switch billing to Yearly?"
        : "Switch billing to Monthly?";
    }
    return pendingCycle === "yearly"
      ? `Switch ${tierLabel} to yearly billing?`
      : `Switch ${tierLabel} to monthly billing?`;
  }

  function getDialogBody(): { label?: string; text: string }[] {
    if (!pendingCycle) return [];
    const periodEndLabel = currentPeriodEndMs
      ? formatShortDate(currentPeriodEndMs)
      : null;
    if (pendingCycle === "yearly") {
      const tierLabel = effectiveTier ? PLAN_LABEL[effectiveTier] : null;
      // `<=` (not `<`) so a misconfigured 0%-savings yearly price still gets
      // the explicit price breakdown rather than falling through to generic
      // copy — the savings line itself is suppressed when savings == 0.
      const hasPrices =
        currentMonthlyCents !== undefined &&
        currentYearlyCents !== undefined &&
        currentMonthlyCents > 0 &&
        currentYearlyCents <= currentMonthlyCents * 12;
      if (!hasPrices) {
        return [
          { text: "You'll be charged the prorated difference today." },
          {
            text:
              currentYearlyCents !== undefined
                ? `Renews at ${formatCents(currentYearlyCents)}/year after this period.`
                : "Renews on the new yearly cadence after this period.",
          },
        ];
      }
      const yearly = formatCents(currentYearlyCents!);
      const monthlyEquivalent = formatCents(
        Math.round(currentYearlyCents! / 12),
      );
      const savingsPercent = Math.round(
        ((currentMonthlyCents! * 12 - currentYearlyCents!) /
          (currentMonthlyCents! * 12)) *
          100,
      );
      // Net charge today = full yearly price minus the unused-monthly credit
      // Stripe will apply. Only surface the exact amount when the backend
      // returned a non-zero credit (admin-granted plans / no Stripe customer
      // return 0 and we fall back to the generic line).
      const netChargeCents = Math.max(
        0,
        currentYearlyCents! - prorationCreditCents,
      );
      const chargedToday =
        prorationCreditCents > 0
          ? {
              label: "Charged today:",
              text: `${formatCents(netChargeCents)} (prorated from your monthly plan).`,
            }
          : { text: "You'll be charged the prorated difference today." };
      const renews = periodEndLabel
        ? { label: "Renews", text: `${yearly}/year on ${periodEndLabel}.` }
        : { label: "Renews", text: `at ${yearly}/year after this period.` };
      return [
        savingsPercent > 0
          ? { text: `Save ${savingsPercent}% with yearly billing.` }
          : null,
        {
          label: tierLabel ? `${tierLabel} yearly:` : "Yearly:",
          text: `${yearly}/year (${monthlyEquivalent}/month).`,
        },
        chargedToday,
        renews,
      ].filter(
        (line): line is { label?: string; text: string } => line !== null,
      );
    }
    const monthly =
      currentMonthlyCents !== undefined ? formatCents(currentMonthlyCents) : "";
    return [
      {
        text: periodEndLabel
          ? `Switches to monthly billing on ${periodEndLabel}.`
          : "Switches to monthly at the end of your current yearly period.",
      },
      monthly ? { label: "New price:", text: `${monthly}/month.` } : null,
      { text: "No charge today." },
    ].filter((line): line is { label?: string; text: string } => line !== null);
  }

  function getTierUpgradeDialogBody(): string {
    if (!pendingTierUpgrade) return "";
    const targetCents =
      serverCycle === "yearly"
        ? tierCostsYearly[pendingTierUpgrade]
        : tierCosts[pendingTierUpgrade];
    const newPrice =
      typeof targetCents === "number" ? formatCents(targetCents) : "";
    const cycleNoun = serverCycle === "yearly" ? "yearly" : "monthly";
    const cycleUnit = serverCycle === "yearly" ? "year" : "month";
    const renewLine = newPrice
      ? `After this period, your plan renews at ${newPrice} per ${cycleUnit}.`
      : "";
    return [
      `You'll be charged the prorated difference immediately for the rest of your ${cycleNoun} period.`,
      renewLine,
    ]
      .filter(Boolean)
      .join(" ");
  }

  async function confirmTierUpgrade() {
    if (!pendingTierUpgrade) return;
    const target = pendingTierUpgrade;
    const ok = await changeTier(target, serverCycle);
    setPendingTierUpgrade(null);
    if (!ok) return;
  }

  function cancelTierUpgrade() {
    setPendingTierUpgrade(null);
  }

  function getTierDowngradeDialogBody(): string {
    if (!pendingTierDowngrade || !plan) return "";
    const targetLabel =
      PLAN_LABEL[pendingTierDowngrade] ?? pendingTierDowngrade;
    const cycleNoun = serverCycle === "yearly" ? "yearly" : "monthly";
    const periodEnd = plan.currentPeriodEnd
      ? formatShortDate(plan.currentPeriodEnd * 1000)
      : null;
    if (periodEnd) {
      return `You'll keep ${plan.label} (${cycleNoun}) until ${periodEnd}, then switch to ${targetLabel}. No charge today.`;
    }
    return `Your plan will switch to ${targetLabel} at the close of the current billing period. No charge today.`;
  }

  async function confirmTierDowngrade() {
    // Capture pendingTierDowngrade locally before the state-clear — passing
    // the captured value to downgradeSubscription guarantees the request
    // targets exactly the tier the user confirmed in the dialog, even if
    // plan.previousTier shifts between dialog open and confirm (e.g. an
    // unrelated subscription refetch lands).
    const target = pendingTierDowngrade;
    if (!target) return;
    setPendingTierDowngrade(null);
    await downgradeSubscription(target);
  }

  function cancelTierDowngrade() {
    setPendingTierDowngrade(null);
  }

  return {
    plan,
    isLoading: subscription.isLoading,
    isUpdatingTier,
    canManagePortal: Boolean(paymentPortal.data),
    // Don't offer upgrade alongside Resume — pending cancel/downgrade should
    // be released first via resumeSubscription, not stacked with a new tier.
    canUpgrade: Boolean(
      plan?.nextTier &&
        !plan?.isPendingCancel &&
        !plan?.isPendingDowngrade &&
        !plan?.isPendingCycleSwitch,
    ),
    // Downgrade only when an active paid sub has a tier below it AND no
    // pending change is already in flight — avoids stacking schedules.
    canDowngrade: Boolean(
      plan?.isPaidPlan &&
        plan?.previousTier &&
        !plan?.isPendingCancel &&
        !plan?.isPendingDowngrade &&
        !plan?.isPendingCycleSwitch,
    ),
    canResume: Boolean(
      plan?.isPendingCancel ||
        plan?.isPendingDowngrade ||
        plan?.isPendingCycleSwitch,
    ),
    selectedCycle,
    pendingCycle,
    pendingTierUpgrade,
    pendingTierUpgradeLabel: pendingTierUpgrade
      ? (PLAN_LABEL[pendingTierUpgrade] ?? pendingTierUpgrade)
      : null,
    pendingTierDowngrade,
    pendingTierDowngradeLabel: pendingTierDowngrade
      ? (PLAN_LABEL[pendingTierDowngrade] ?? pendingTierDowngrade)
      : null,
    isCycleToggleVisible,
    cycleDialogTitle: getDialogTitle(),
    cycleDialogBody: getDialogBody(),
    tierUpgradeDialogBody: getTierUpgradeDialogBody(),
    tierDowngradeDialogBody: getTierDowngradeDialogBody(),
    onCycleChange,
    onConfirmCycleSwitch: () => {
      void confirmCycleSwitch();
    },
    onCancelCycleSwitch: cancelCycleSwitch,
    onConfirmTierUpgrade: () => {
      void confirmTierUpgrade();
    },
    onCancelTierUpgrade: cancelTierUpgrade,
    onConfirmTierDowngrade: () => {
      void confirmTierDowngrade();
    },
    onCancelTierDowngrade: cancelTierDowngrade,
    onUpgrade: () => {
      if (!plan?.nextTier) return;
      // Team (BUSINESS) tier is contact-sales — divert to marketing page
      // instead of POSTing a Checkout the user can't self-serve.
      if (plan.nextTierIsTeamLink) {
        window.open(TEAM_UPGRADE_URL, "_blank", "noopener,noreferrer");
        return;
      }
      // Free tier (NO_TIER) flow runs through Stripe Checkout via the
      // success_url redirect — there's no immediate proration risk, so go
      // straight to changeTier without a confirm step. Only paid→paid
      // upgrades hit the always_invoice immediate-billing path.
      if (!plan.isPaidPlan) {
        void changeTier(plan.nextTier, selectedCycle);
        return;
      }
      setPendingTierUpgrade(plan.nextTier);
    },
    onDowngrade: () => {
      if (!plan?.previousTier) return;
      setPendingTierDowngrade(plan.previousTier);
    },
    onResume: () => {
      void resumeSubscription();
    },
    onManage: () => {
      if (paymentPortal.data) window.location.href = paymentPortal.data;
    },
  };
}
