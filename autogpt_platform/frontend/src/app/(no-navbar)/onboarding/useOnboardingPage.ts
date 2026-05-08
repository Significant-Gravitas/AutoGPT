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
import { normalizeOnboardingProfile } from "./helpers";
import { Step, useOnboardingWizardStore } from "./store";

const LD_INIT_TIMEOUT_SECONDS = 5;

// SessionStorage ceiling for the wizard. The backend's `completedSteps`
// only records VISIT_COPILOT at the very end (the 5 in-wizard steps aren't
// tracked individually), so resume / fast-forward guardrails are enforced
// client-side: the user can only land on a step they've previously reached.
const STEP_STORAGE_KEY = "autogpt:onboarding-highest-step";

function parseStepParam(value: string | null, maxStep: number): Step | null {
  const n = Number(value);
  if (Number.isInteger(n) && n >= 1 && n <= maxStep) return n as Step;
  return null;
}

function readHighestStep(): number {
  if (typeof window === "undefined") return 1;
  const raw = window.sessionStorage.getItem(STEP_STORAGE_KEY);
  const n = Number(raw);
  return Number.isInteger(n) && n >= 1 ? n : 1;
}

function writeHighestStep(step: number) {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(STEP_STORAGE_KEY, String(step));
}

function clearHighestStep() {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(STEP_STORAGE_KEY);
}

export function useOnboardingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isLoggedIn } = useSupabase();
  const currentStep = useOnboardingWizardStore((s) => s.currentStep);
  const goToStep = useOnboardingWizardStore((s) => s.goToStep);

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

  // Initialise store from URL on mount, clamp ?step= to the highest step
  // the user has actually reached. No-step URL resumes from the highest
  // reached so refreshes don't drop the user back to step 1. Form data is
  // persisted (zustand persist), so we no longer reset on mount — that
  // would defeat the point of persistence by wiping the user's name/role
  // every time they refresh mid-wizard.
  useEffect(() => {
    if (!areFlagsReady || hasInitialized.current) return;
    hasInitialized.current = true;
    const urlStep = parseStepParam(searchParams.get("step"), preparingStep);
    // A successful Stripe checkout return is a trusted intent to advance to
    // Preparing — without this, the highestStep ceiling (capped at 4 before
    // redirect) clamps the user back to step 4 and onboarding deadlocks.
    const isSubscriptionSuccess =
      searchParams.get("subscription") === "success";
    const ceiling = isSubscriptionSuccess
      ? preparingStep
      : (Math.min(readHighestStep(), preparingStep) as Step);
    const target = (
      urlStep === null ? ceiling : Math.min(urlStep, ceiling)
    ) as Step;
    goToStep(target);
  }, [areFlagsReady, searchParams, goToStep, preparingStep]);

  // Sync store → URL when step changes; record the new ceiling.
  useEffect(() => {
    if (!areFlagsReady) return;
    const urlStep = parseStepParam(searchParams.get("step"), preparingStep);
    if (currentStep !== urlStep) {
      router.replace(`/onboarding?step=${currentStep}`, { scroll: false });
    }
    if (currentStep > readHighestStep()) {
      writeHighestStep(currentStep);
    }
  }, [areFlagsReady, currentStep, router, searchParams, preparingStep]);

  // Check if onboarding already completed
  useEffect(() => {
    if (!isLoggedIn) return;

    async function checkCompletion() {
      try {
        const onboarding = await resolveResponse(getV1OnboardingState());
        if (onboarding.completedSteps.includes("VISIT_COPILOT")) {
          clearHighestStep();
          // Clear the persisted form data without touching in-memory state.
          // `reset()` would set currentStep=1 and trip the URL-sync effect
          // into racing with the /copilot redirect (the spurious
          // /onboarding?step=1 replace wins, stranding the user on Welcome
          // until they refresh).
          useOnboardingWizardStore.persist.clearStorage();
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

    const { name, role, painPoints } = normalizeOnboardingProfile(
      useOnboardingWizardStore.getState(),
    );

    // Profile is submitted before the Stripe Checkout redirect (defence in
    // depth: persist keeps the store across the round-trip, but submitting
    // pre-redirect ensures the backend has the data even if the user
    // closes the tab during checkout). Skip the resubmit on return to
    // avoid overwriting saved data with empties — belt-and-suspenders:
    // also skip when `name` is empty so that even if the success query
    // param gets stripped (manual edit, share link, upstream proxy
    // normalising URLs), we don't blank the saved profile.
    if (searchParams.get("subscription") === "success" || !name.trim()) return;

    postV1SubmitOnboardingProfile({
      user_name: name,
      user_role: role,
      pain_points: painPoints,
    }).catch(() => {
      // Best effort — profile data is non-critical for accessing copilot
    });
  }, [currentStep, preparingStep, searchParams]);

  async function handlePreparingComplete() {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await postV1CompleteOnboardingStep({ step: "VISIT_COPILOT" });
        clearHighestStep();
        useOnboardingWizardStore.persist.clearStorage();
        router.replace("/copilot");
        return;
      } catch {
        if (attempt < 2) await new Promise((r) => setTimeout(r, 1000));
      }
    }
    clearHighestStep();
    useOnboardingWizardStore.persist.clearStorage();
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
