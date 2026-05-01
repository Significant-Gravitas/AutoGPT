import {
  getV1OnboardingState,
  postV1CompleteOnboardingStep,
  postV1SubmitOnboardingProfile,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { resolveResponse } from "@/app/api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useLDClient } from "launchdarkly-react-client-sdk";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Step, useOnboardingWizardStore } from "./store";

const LD_INIT_TIMEOUT_SECONDS = 5;

function parseStep(value: string | null, maxStep: number): Step {
  const n = Number(value);
  if (Number.isInteger(n) && n >= 1 && n <= maxStep) return n as Step;
  return 1;
}

export function useOnboardingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isLoggedIn } = useSupabase();
  const currentStep = useOnboardingWizardStore((s) => s.currentStep);
  const goToStep = useOnboardingWizardStore((s) => s.goToStep);
  const reset = useOnboardingWizardStore((s) => s.reset);

  // Wait for LaunchDarkly before initialising the wizard from the URL.
  // Without this, the init effect runs against the default flag value
  // (false) on first render and clamps e.g. ?step=5 down to step 1; once
  // the flag resolves to true, the hasInitialized guard blocks re-init
  // and the user is stuck on step 1.
  const ldClient = useLDClient();
  const ldEnabled = environment.areFeatureFlagsEnabled();
  const [areFlagsReady, setAreFlagsReady] = useState(!ldEnabled);
  useEffect(() => {
    if (!ldEnabled || !ldClient || areFlagsReady) return;
    let cancelled = false;
    // Use the same 5s timeout as LDProvider so the wizard never hangs
    // when LaunchDarkly is unreachable; on timeout we fall back to the
    // default flag values that useGetFlag already returns.
    ldClient
      .waitForInitialization(LD_INIT_TIMEOUT_SECONDS)
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setAreFlagsReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, [ldClient, ldEnabled, areFlagsReady]);

  // Snapshot the flag once LaunchDarkly resolves so an admin toggling
  // ENABLE_PLATFORM_PAYMENT mid-session can't shuffle steps under a user
  // who is already inside the wizard.
  const livePaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const paymentEnabledSnapshot = useRef<boolean | null>(null);
  if (paymentEnabledSnapshot.current === null && areFlagsReady) {
    paymentEnabledSnapshot.current = livePaymentEnabled;
  }
  const isPaymentEnabled = paymentEnabledSnapshot.current ?? false;
  const preparingStep: Step = isPaymentEnabled ? 5 : 4;
  const totalSteps = isPaymentEnabled ? 4 : 3;

  const [isOnboardingStateLoading, setIsOnboardingStateLoading] =
    useState(true);
  const hasSubmitted = useRef(false);
  const hasInitialized = useRef(false);

  // Initialise store from URL on mount, reset form data
  useEffect(() => {
    if (!areFlagsReady || hasInitialized.current) return;
    hasInitialized.current = true;
    const urlStep = parseStep(searchParams.get("step"), preparingStep);
    reset();
    goToStep(urlStep);
  }, [areFlagsReady, searchParams, reset, goToStep, preparingStep]);

  // Sync store → URL when step changes
  useEffect(() => {
    if (!areFlagsReady) return;
    const urlStep = parseStep(searchParams.get("step"), preparingStep);
    if (currentStep !== urlStep) {
      router.replace(`/onboarding?step=${currentStep}`, { scroll: false });
    }
  }, [areFlagsReady, currentStep, router, searchParams, preparingStep]);

  // Check if onboarding already completed
  useEffect(() => {
    if (!isLoggedIn) return;

    async function checkCompletion() {
      try {
        const onboarding = await resolveResponse(getV1OnboardingState());
        if (onboarding.completedSteps.includes("VISIT_COPILOT")) {
          router.replace("/copilot");
          return;
        }
      } catch {
        // If we can't check, show onboarding anyway
      }
      setIsOnboardingStateLoading(false);
    }

    checkCompletion();
  }, [isLoggedIn, router]);

  // Submit profile when entering the Preparing step
  useEffect(() => {
    if (currentStep !== preparingStep || hasSubmitted.current) return;
    hasSubmitted.current = true;

    const { name, role, otherRole, painPoints, otherPainPoint } =
      useOnboardingWizardStore.getState();
    const resolvedRole = role === "Other" ? otherRole : role;
    const resolvedPainPoints = painPoints
      .filter((p) => p !== "Something else")
      .concat(
        painPoints.includes("Something else") && otherPainPoint.trim()
          ? [otherPainPoint.trim()]
          : [],
      );

    postV1SubmitOnboardingProfile({
      user_name: name,
      user_role: resolvedRole,
      pain_points: resolvedPainPoints,
    }).catch(() => {
      // Best effort — profile data is non-critical for accessing copilot
    });
  }, [currentStep, preparingStep]);

  async function handlePreparingComplete() {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await postV1CompleteOnboardingStep({ step: "VISIT_COPILOT" });
        router.replace("/copilot");
        return;
      } catch {
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000));
      }
    }
    router.replace("/copilot");
  }

  return {
    currentStep,
    isLoading: isOnboardingStateLoading || !areFlagsReady,
    handlePreparingComplete,
    isPaymentEnabled,
    preparingStep,
    totalSteps,
  };
}
