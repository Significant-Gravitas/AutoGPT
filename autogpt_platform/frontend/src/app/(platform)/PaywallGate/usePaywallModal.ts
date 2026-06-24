"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  useGetSubscriptionStatus,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useMountEffect } from "@/hooks/useMountEffect";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
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

const PLAN_LABEL: Record<string, string> = {
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
};

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
  const {
    data: subscription,
    isLoading,
    isFetching,
    refetch: refetchSubscription,
  } = useGetSubscriptionStatus({
    query: { select: (res) => (res.status === 200 ? res.data : null) },
  });
  const { mutateAsync: updateTier, isPending } = useUpdateSubscriptionTier();
  const { validateSession } = useSupabase();
  const router = useRouter();

  // The paywall can outlive the session that mounted it (failed logout,
  // history restore, expired cookies). Re-check the session server-side on
  // mount: an invalid one clears the user — unmounting the gate — and
  // redirects to /login instead of leaving a zombie paywall whose checkout
  // calls can only 401.
  useMountEffect(() => {
    void validateSession();
  });
  const [selectedCycle, setSelectedCycle] = useState<"monthly" | "yearly">(
    "monthly",
  );
  const [selectedTier, setSelectedTier] = useState<string | null>(null);
  // When the user already has an active Stripe subscription (admin override
  // flipped DB tier to NO_TIER but Stripe sub is still live), modify-in-place
  // skips Stripe Checkout entirely — clicking Upgrade silently re-bills the
  // saved card and looks like a no-op. Stage the target tier here and surface
  // a confirmation dialog before firing the mutation.
  const [pendingTier, setPendingTier] = useState<string | null>(null);

  const country = COUNTRIES[0];
  const isYearly = selectedCycle === "yearly";

  const plans = deriveAvailablePlans(
    subscription?.tier_costs,
    subscription?.tier_costs_yearly,
  );

  const hasActiveStripeSubscription = Boolean(
    subscription?.has_active_stripe_subscription,
  );

  async function fireUpdate(tier: string) {
    setSelectedTier(tier);
    try {
      const result = await updateTier({
        data: {
          tier: tier as SubscriptionTierRequestTier,
          success_url: `${window.location.origin}/profile/credits?subscription=success`,
          // Backing out of Stripe Checkout must return to the page the user
          // was on so the paywall re-gates immediately. /profile/* is
          // paywall-exempt — landing there let users wander an app that
          // looks unlocked until the next non-exempt navigation.
          cancel_url: window.location.href,
          billing_cycle: isYearly ? "yearly" : "monthly",
        },
      });
      const url = (result?.data as CheckoutResponse | undefined)?.url;
      if (url) {
        window.location.href = url;
      }
    } catch (error) {
      // 401 means the paywall outlived its session: checkout can never
      // succeed, so route to login instead of toasting a dead-end error.
      if (error instanceof ApiError && error.status === 401) {
        router.replace("/login");
        return;
      }
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
    // Active Stripe sub → modify-in-place hits saved-card auto-charge; no
    // Stripe Checkout opens, so the click would otherwise be a silent
    // re-bill / refund. Stage the tier behind the confirmation dialog.
    if (hasActiveStripeSubscription) {
      setPendingTier(tier);
      return;
    }
    await fireUpdate(tier);
  }

  async function confirmPendingTier() {
    if (!pendingTier) return;
    const tier = pendingTier;
    await fireUpdate(tier);
    setPendingTier(null);
  }

  function cancelPendingTier() {
    setPendingTier(null);
  }

  // Route through the shared /logout page (same path the Navbar uses) rather
  // than calling logOut() inline: it runs the full sign-out, then redirects to
  // /login. Navigating away also unmounts this modal — without the redirect the
  // user would be left logged-out but stranded behind the paywall.
  function handleLogout() {
    router.replace("/logout");
  }

  const pendingTierLabel = pendingTier
    ? (PLAN_LABEL[pendingTier] ?? pendingTier)
    : null;

  function retryLoadPlans() {
    void refetchSubscription();
  }

  return {
    isLoading,
    plans,
    retryLoadPlans,
    isRetryingPlans: isFetching,
    country,
    isYearly,
    selectedCycle,
    setSelectedCycle,
    handleSelectPlan,
    isPending,
    selectedTier,
    pendingTier,
    pendingTierLabel,
    confirmPendingTier,
    cancelPendingTier,
    handleLogout,
  };
}
