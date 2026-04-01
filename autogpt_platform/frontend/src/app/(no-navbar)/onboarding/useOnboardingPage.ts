import {
  getV1OnboardingState,
  postV1CompleteOnboardingStep,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import { resolveResponse } from "@/app/api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "./store";

export function useOnboardingPage() {
  const router = useRouter();
  const { isLoggedIn } = useSupabase();
  const currentStep = useOnboardingWizardStore((s) => s.currentStep);
  const name = useOnboardingWizardStore((s) => s.name);
  const role = useOnboardingWizardStore((s) => s.role);
  const otherRole = useOnboardingWizardStore((s) => s.otherRole);
  const painPoints = useOnboardingWizardStore((s) => s.painPoints);
  const otherPainPoint = useOnboardingWizardStore((s) => s.otherPainPoint);
  const [isLoading, setIsLoading] = useState(true);
  const hasSubmitted = useRef(false);

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
      setIsLoading(false);
    }

    checkCompletion();
  }, [isLoggedIn, router]);

  // Submit profile when entering step 4
  useEffect(() => {
    if (currentStep !== 4 || hasSubmitted.current) return;
    hasSubmitted.current = true;

    const resolvedRole = role === "Other" ? otherRole : role;
    const resolvedPainPoints = painPoints
      .filter((p) => p !== "Something else")
      .concat(
        painPoints.includes("Something else") && otherPainPoint.trim()
          ? [otherPainPoint.trim()]
          : [],
      );

    // Fire and forget — the preparing screen waits 4s regardless
    customMutator("/api/onboarding/profile", {
      method: "POST",
      body: JSON.stringify({
        user_name: name,
        user_role: resolvedRole,
        pain_points: resolvedPainPoints,
      }),
    }).catch(() => {
      // Best effort — profile data is non-critical for accessing copilot
    });
  }, [currentStep, name, role, otherRole, painPoints, otherPainPoint]);

  const handlePreparingComplete = useCallback(async () => {
    try {
      await postV1CompleteOnboardingStep({ step: "VISIT_COPILOT" });
    } catch {
      // Best effort
    }
    router.replace("/copilot");
  }, [router]);

  return {
    currentStep,
    isLoading,
    handlePreparingComplete,
  };
}
