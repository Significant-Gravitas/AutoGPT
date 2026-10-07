"use client";
import { useRouter } from "next/navigation";
import { useProActivation } from "@/services/pro-activation/useProActivation";
import { useUsageExperience } from "./useUsageExperience";

export function useUsageActions() {
  const state = useUsageExperience();
  const router = useRouter();
  const activation = useProActivation();
  const acceptedProTrial =
    state.experience.isActiveTrial && state.trial?.offer?.tier === "PRO";
  const target = state.experience.targetTier;
  const cycle: "monthly" | "yearly" = acceptedProTrial
    ? (state.trial?.offer?.billing_cycle ?? "monthly")
    : state.subscription?.billing_cycle === "yearly"
      ? "yearly"
      : "monthly";
  const prices =
    cycle === "yearly"
      ? state.subscription?.tier_costs_yearly
      : state.subscription?.tier_costs;
  const offer = {
    tier: target ?? ("PRO" as const),
    price: acceptedProTrial
      ? state.trial?.offer?.unit_amount
      : target
        ? prices?.[target]
        : undefined,
    currency: acceptedProTrial ? state.trial?.offer?.currency : "USD",
    cycle,
    freshUsage: acceptedProTrial,
    usageMultiplier:
      state.subscription?.tier_multipliers?.MAX &&
      state.subscription?.tier_multipliers?.PRO
        ? Math.round(
            (state.subscription.tier_multipliers.MAX /
              state.subscription.tier_multipliers.PRO) *
              10,
          ) / 10
        : undefined,
    disabled: activation.isBusy,
  };
  function upgrade() {
    if (acceptedProTrial) {
      void activation.start(
        window.location.pathname +
          window.location.search +
          window.location.hash,
      );
    } else {
      router.push(
        target === "MAX"
          ? "/settings/billing?upgrade=MAX"
          : "/settings/billing",
      );
    }
  }
  return { ...state, offer, upgrade };
}
