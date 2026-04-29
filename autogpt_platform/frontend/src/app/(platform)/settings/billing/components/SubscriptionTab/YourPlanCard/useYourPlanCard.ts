"use client";

import {
  useGetSubscriptionStatus,
  useGetV1ManagePaymentMethods,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { GetV1ManagePaymentMethods200 } from "@/app/api/__generated__/models/getV1ManagePaymentMethods200";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";

const PLAN_LABEL: Record<string, string> = {
  BASIC: "Basic",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Business",
  ENTERPRISE: "Enterprise",
};

const TIER_ORDER = ["BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"] as const;

function getNextTier(current: string): string | null {
  const idx = TIER_ORDER.indexOf(current as (typeof TIER_ORDER)[number]);
  if (idx === -1 || idx === TIER_ORDER.length - 1) return null;
  return TIER_ORDER[idx + 1];
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

  const plan = subscription.data
    ? {
        tierKey: subscription.data.tier,
        label: PLAN_LABEL[subscription.data.tier] ?? subscription.data.tier,
        monthlyCostCents: subscription.data.monthly_cost,
        isPaidPlan: subscription.data.tier !== "BASIC",
        nextTier: getNextTier(subscription.data.tier),
      }
    : undefined;

  async function changeTier(tier: string) {
    const successUrl = `${window.location.origin}${window.location.pathname}?subscription=success`;
    const cancelUrl = `${window.location.origin}${window.location.pathname}?subscription=cancelled`;
    const result = await updateTier({
      data: {
        tier: tier as SubscriptionTierRequestTier,
        success_url: successUrl,
        cancel_url: cancelUrl,
      },
    });
    const url = (result?.data as { url?: string } | undefined)?.url;
    if (url) window.location.href = url;
    await subscription.refetch();
  }

  return {
    plan,
    isLoading: subscription.isLoading,
    isUpdatingTier,
    canManagePortal: Boolean(paymentPortal.data),
    canUpgrade: Boolean(plan?.nextTier),
    onUpgrade: () => {
      if (plan?.nextTier) void changeTier(plan.nextTier);
    },
    onCancel: () => void changeTier("BASIC"),
    onManage: () => {
      if (paymentPortal.data) window.location.href = paymentPortal.data;
    },
  };
}
