import React from "react";
import { fixture, usage, trial, tiers, noop, upgradeReview } from "./fixtures";
import { PLAN_METADATA } from "@/components/molecules/PlanCard/plans";
import { COUNTRIES } from "@/components/molecules/PlanCard/countries";
import { getUsageExperience } from "@/services/usageExperience/helpers";

export function useWorkspaceStorage() {
  return { data: { used_bytes: 0, limit_bytes: 262144000, file_count: 0 } };
}
export function useTrialStatus() {
  return { data: trial(), isLoading: false };
}
export function useTrialCheckoutReturn() {
  return { ready: true, error: null, retry: noop };
}
export function useTrialCard() {
  return {
    userID: "fixture-user",
    trial: trial(),
    isLoading: false,
    queryError: false,
    error: null,
    retry: noop,
    isStarting: false,
    isCanceling: false,
    isResuming: false,
    resumeTrial: noop,
    showCanceledDialog: false,
    dismissCanceledDialog: noop,
    subscribeNow: noop,
    startTrial: async () => {},
    cancelTrial: async () => {},
  };
}
export function usePaymentMethodCard() {
  return { canManage: true, isOpening: false, onManage: noop };
}
export function useInvoicesCard() {
  return { invoices: [], isLoading: false, isError: false, refetch: noop };
}
export function useYourPlanCard() {
  const ended =
    ["expired", "canceled"].includes(fixture.state) ||
    fixture.state === "payment_failed";
  const tier = ended ? "NO_TIER" : tiers[fixture.tier];
  const labels: Record<string, string> = {
    NO_TIER: "No active subscription",
    PRO: "Pro",
    MAX: "Max",
    BUSINESS: "Team",
    BASIC: "Basic",
    ENTERPRISE: "Enterprise",
  };
  const order = ["PRO", "MAX", "BUSINESS"];
  const i = order.indexOf(tier);
  const next =
    tier === "ENTERPRISE" || i === 2 ? null : i === -1 ? "PRO" : order[i + 1];
  const previous = i > 0 ? order[i - 1] : null;
  const isPaid = tier !== "NO_TIER";
  const isCycleToggleVisible =
    tier !== null &&
    tier !== "ENTERPRISE" &&
    tier !== "BASIC" &&
    tier !== "NO_TIER";
  return {
    upgradeUsageMultiplier: 8.5,
    upgradeReview,
    plan: {
      tierKey: tier,
      label: labels[tier] ?? tier,
      monthlyCostCents:
        tier === "PRO"
          ? 5000
          : tier === "MAX"
            ? 32000
            : tier === "BUSINESS"
              ? 100000
              : tier === "BASIC"
                ? 1000
                : 0,
      isPaidPlan: isPaid,
      nextTier: next,
      nextTierLabel: next ? (labels[next] ?? next) : null,
      nextTierIsTeamLink: next === "BUSINESS",
      previousTier: previous,
      previousTierLabel: previous ? labels[previous] : null,
      currentPeriodEnd: new Date("2026-11-05T12:00:00Z").getTime() / 1000,
      pendingTier: null,
      pendingTierLabel: null,
      pendingEffectiveAt: null,
      pendingCycle: null,
      isPendingCancel: false,
      isPendingDowngrade: false,
      isPendingCycleSwitch: false,
      billingCycle: "monthly",
      isAdminManaged: tier === "ENTERPRISE",
    },
    offerPriceCents: next === "PRO" ? 5000 : next === "MAX" ? 32000 : undefined,
    isError: false,
    retry: noop,
    isLoading: false,
    isUpdatingTier: false,
    canManagePortal: true,
    canUpgrade: !!next,
    canDowngrade: isPaid && !!previous,
    canResume: false,
    selectedCycle: "monthly",
    pendingCycle: null,
    pendingTierUpgrade: null,
    pendingTierUpgradeLabel: null,
    pendingTierDowngrade: null,
    pendingTierDowngradeLabel: null,
    isCycleToggleVisible,
    cycleDialogTitle: "",
    cycleDialogBody: [],
    tierUpgradeDialogBody: "",
    tierDowngradeDialogBody: "",
    onCycleChange: noop,
    onConfirmCycleSwitch: noop,
    onCancelCycleSwitch: noop,
    onConfirmTierUpgrade: noop,
    onCancelTierUpgrade: noop,
    onConfirmTierDowngrade: noop,
    onCancelTierDowngrade: noop,
    onUpgrade: noop,
    onDowngrade: noop,
    onResume: noop,
    onManage: noop,
  };
}
export function usePaywallModal() {
  return {
    isLoading: false,
    plans: [
      { ...PLAN_METADATA.PRO, usdMonthly: 50, usdYearly: 510 },
      { ...PLAN_METADATA.MAX, usdMonthly: 320, usdYearly: 3264 },
    ],
    retryLoadPlans: noop,
    isRetryingPlans: false,
    country: COUNTRIES[0],
    selectedCycle: "monthly",
    setSelectedCycle: noop,
    handleSelectPlan: noop,
    isPending: false,
    selectedTier: null,
    pendingTier: null,
    pendingTierLabel: null,
    confirmPendingTier: noop,
    cancelPendingTier: noop,
    handleLogout: noop,
    handleClose: noop,
    trial: useTrialCard(),
    trialOffer: null,
  };
}
export function useRouter() {
  return { push: noop, replace: noop };
}
export const usePathname = () => "/settings/billing";
export const useSearchParams = () => new URLSearchParams();
export default React.forwardRef<HTMLAnchorElement, React.ComponentProps<"a">>(
  function Link({ children, ...props }, ref) {
    return (
      <a ref={ref} {...props}>
        {children}
      </a>
    );
  },
);

export const ENTERPRISE_CONTACT_URL = "mailto:contact@agpt.co";
export const TEAM_UPGRADE_URL = "https://tally.so/r/2Eb9zj";
export function useProActivation() {
  return { start: noop, isBusy: false, isReady: fixture.state === "fresh" };
}
export function useUsageExperience() {
  const u = usage();
  const t = trial();
  return {
    usage: u,
    trial: t,
    experience: getUsageExperience(u, t, useYourPlanCard().plan.tierKey),
    subscription: {
      tier: useYourPlanCard().plan.tierKey,
      billing_cycle: "monthly",
      tier_costs: { PRO: 5000, MAX: 32000 },
      tier_multipliers: { PRO: 1, MAX: 8.5 },
    },
    isLoading: false,
    isError: fixture.state === "unavailable",
    retry: noop,
    isBillingEnabled: true,
  };
}

export const captureException = noop;
export const captureMessage = noop;
export const withScope = noop;

export function usePatchV1FulfillCheckoutSession() {
  return { mutateAsync: async () => {} };
}
export const getGetSubscriptionStatusQueryKey = () => ["subscription"];
export function AutomationCreditsTab() {
  return null;
}
export const trackCheckoutAbandoned = noop;
export const trackTrialCheckoutAbandoned = noop;

// The review links to the real payment management component; only its API reply is fixture data.
export function useGetV1ManagePaymentMethods() {
  return {
    data: {
      status: 200,
      data: { url: "https://billing.stripe.com/p/session/fixture" },
    },
    isLoading: false,
  };
}

export function useGetSubscriptionStatus(options = {}) {
  const response = {
    status: 200,
    data: {
      tier: useYourPlanCard().plan.tierKey,
      tier_costs: { PRO: 5000, MAX: 32000 },
      tier_multipliers: { PRO: 1, MAX: 8.5 },
    },
  };
  const data = options.query?.select
    ? options.query.select(response)
    : response;
  return { data, isPending: false, isLoading: false };
}
export function notFound() {
  throw new Error("Unexpected unavailable fixture route");
}

export const getGetV1ListStripeInvoicesQueryKey = () => ["invoices"];
export const trackPaywallViewed = noop;
export const trackPlanSelected = noop;
export const addBreadcrumb = noop;
export const getTraceData = () => ({});
export function useUpdateSubscriptionTier() {
  return { mutateAsync: async () => {}, isPending: false };
}

export const getGetTrialsGetTrialStatusQueryKey = () => ["trial"];

export const getGetV2GetCopilotUsageQueryKey = () => ["usage"];
export class ApiError extends Error {}

export function useGetV2ListChatTransports() {
  return {
    data: {
      status: 200,
      data: {
        transports: [
          { available: true, auth_provider: "platform", credential_id: null },
        ],
      },
    },
  };
}
