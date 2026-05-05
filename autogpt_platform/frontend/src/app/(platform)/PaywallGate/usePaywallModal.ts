"use client";

import { useState } from "react";
import {
  useGetSubscriptionStatus,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { toast } from "@/components/molecules/Toast/use-toast";
import { COUNTRIES } from "@/components/molecules/PlanCard/countries";
import {
  type BackendTierKey,
  PLAN_METADATA,
  type PlanDef,
  TEAM_INTAKE_FORM_URL,
} from "@/components/molecules/PlanCard/plans";

interface CheckoutResponse {
  url?: string;
}

const TIER_DISPLAY_ORDER: BackendTierKey[] = ["PRO", "MAX", "BUSINESS"];

// Build a PlanDef per priceable tier exposed by /credits/subscription. Static
// metadata (features / cta / highlight) is layered on top of the API-driven
// monthly + yearly amounts, so a tier appears here iff LD has a Stripe price
// for it.
function deriveAvailablePlans(
  tierCosts: Record<string, number> | undefined,
  tierCostsYearly: Record<string, number> | undefined,
): PlanDef[] {
  if (!tierCosts) return [];
  return TIER_DISPLAY_ORDER.filter(
    (tier) => typeof tierCosts[tier] === "number",
  ).map((tier) => {
    const monthlyCents = tierCosts[tier] ?? 0;
    const yearlyCents = tierCostsYearly?.[tier];
    return {
      ...PLAN_METADATA[tier],
      usdMonthly: monthlyCents / 100,
      usdYearly: typeof yearlyCents === "number" ? yearlyCents / 100 : null,
    };
  });
}

export function usePaywallModal() {
  const { data: subscription, isLoading } = useGetSubscriptionStatus({
    query: { select: (res) => (res.status === 200 ? res.data : null) },
  });
  const { mutateAsync: updateTier, isPending } = useUpdateSubscriptionTier();
  const [selectedCycle, setSelectedCycle] = useState<"monthly" | "yearly">(
    "monthly",
  );
  const [selectedTier, setSelectedTier] = useState<string | null>(null);

  const country = COUNTRIES[0];
  const isYearly = selectedCycle === "yearly";

  const plans = deriveAvailablePlans(
    subscription?.tier_costs,
    subscription?.tier_costs_yearly,
  );

  async function handleSelectPlan(tier: string) {
    if (isPending) return;
    // Team (BUSINESS) is contact-sales, not a self-serve Stripe Checkout —
    // divert to the intake form like onboarding + Settings billing do.
    // Without this, the POST hits the backend with tier=BUSINESS which 422s
    // (no LD price) and surfaces a generic "couldn't start checkout" toast.
    if (tier === "BUSINESS") {
      window.open(TEAM_INTAKE_FORM_URL, "_blank", "noopener,noreferrer");
      return;
    }
    setSelectedTier(tier);
    try {
      const baseUrl = `${window.location.origin}/profile/credits`;
      const result = await updateTier({
        data: {
          tier: tier as SubscriptionTierRequestTier,
          success_url: `${baseUrl}?subscription=success`,
          cancel_url: `${baseUrl}?subscription=cancelled`,
          billing_cycle: isYearly ? "yearly" : "monthly",
        },
      });
      const url = (result?.data as CheckoutResponse | undefined)?.url;
      if (url) {
        window.location.href = url;
      }
    } catch (error) {
      toast({
        title: "Couldn't start checkout",
        description:
          error instanceof Error
            ? error.message
            : "Stripe didn't accept the request. Please try again.",
        variant: "destructive",
      });
    } finally {
      setSelectedTier(null);
    }
  }

  return {
    isLoading,
    plans,
    country,
    isYearly,
    selectedCycle,
    setSelectedCycle,
    handleSelectPlan,
    isPending,
    selectedTier,
  };
}
