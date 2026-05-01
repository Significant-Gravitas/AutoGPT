import { useUpdateSubscriptionTier } from "@/app/api/__generated__/endpoints/credits/credits";
import { postV1SubmitOnboardingProfile } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import { normalizeOnboardingProfile } from "../../helpers";
import { useOnboardingWizardStore } from "../../store";
import { COUNTRIES } from "./countries";
import { PLAN_KEYS, type PlanKey, TEAM_INTAKE_FORM_URL } from "./helpers";

const PLAN_TO_TIER: Record<
  Exclude<PlanKey, typeof PLAN_KEYS.TEAM>,
  SubscriptionTierRequestTier
> = {
  PRO: "PRO",
  MAX: "MAX",
};

interface CheckoutResponse {
  url?: string;
}

export function useSubscriptionStep() {
  const billing = useOnboardingWizardStore((s) => s.selectedBilling);
  const setSelectedBilling = useOnboardingWizardStore(
    (s) => s.setSelectedBilling,
  );
  const selectedCountryCode = useOnboardingWizardStore(
    (s) => s.selectedCountryCode,
  );
  const setSelectedCountryCode = useOnboardingWizardStore(
    (s) => s.setSelectedCountryCode,
  );
  const setSelectedPlan = useOnboardingWizardStore((s) => s.setSelectedPlan);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const selectedPlan = useOnboardingWizardStore((s) => s.selectedPlan);

  const { mutateAsync: updateTier, isPending: isUpdatingTier } =
    useUpdateSubscriptionTier();
  // Local guard that flips synchronously on first click so the profile-save
  // phase (which runs before `isUpdatingTier` becomes true) can't be
  // re-entered by a fast double-click queueing duplicate POSTs.
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isProcessing = isUpdatingTier || isSubmitting;

  const countryIdx = Math.max(
    0,
    COUNTRIES.findIndex((c) => c.countryCode === selectedCountryCode),
  );
  const country = COUNTRIES[countryIdx];
  const isYearly = billing === "yearly";

  function setCountryIdx(idx: number) {
    const next = COUNTRIES[idx];
    if (!next) return;
    setSelectedCountryCode(next.countryCode);
  }

  async function handlePlanSelect(planKey: PlanKey) {
    if (planKey === PLAN_KEYS.TEAM) {
      window.open(TEAM_INTAKE_FORM_URL, "_blank", "noopener,noreferrer");
      return;
    }
    if (isProcessing) return;
    setIsSubmitting(true);

    setSelectedPlan(planKey);
    const tier = PLAN_TO_TIER[planKey];

    // Profile data lives in an in-memory zustand store, so persist it before
    // Stripe takes the user off-page — otherwise the user's name/role/pain
    // points are lost on the post-checkout redirect.
    const { name, role, painPoints } = normalizeOnboardingProfile(
      useOnboardingWizardStore.getState(),
    );

    try {
      // Profile must be persisted before Stripe takes over — the zustand
      // store is in-memory and the post-checkout return is a full page
      // navigation. Abort on failure rather than silently losing the user's
      // name / role / pain points.
      await postV1SubmitOnboardingProfile({
        user_name: name,
        user_role: role,
        pain_points: painPoints,
      });

      const baseUrl = `${window.location.origin}/onboarding`;
      const result = await updateTier({
        data: {
          tier,
          success_url: `${baseUrl}?step=5&subscription=success`,
          cancel_url: `${baseUrl}?step=4&subscription=cancelled`,
        },
      });
      const url = (result?.data as CheckoutResponse | undefined)?.url;
      if (url) {
        // Navigating away — don't refetch (would set state on an
        // unmounting component while Stripe Checkout takes over).
        window.location.href = url;
        return;
      }
      // Backend modified the subscription in place (no Checkout URL) —
      // proceed to the preparing step as if the user had clicked through.
      nextStep();
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
      setIsSubmitting(false);
    }
  }

  return {
    billing,
    setBilling: setSelectedBilling,
    countryIdx,
    setCountryIdx,
    country,
    isYearly,
    handlePlanSelect,
    isUpdatingTier: isProcessing,
    selectedPlan,
  };
}
