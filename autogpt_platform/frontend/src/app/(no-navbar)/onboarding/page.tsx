"use client";
import { getV1OnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { getOnboardingStatus, resolveResponse } from "@/app/api/helpers";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function OnboardingPage() {
  const router = useRouter();

  useEffect(() => {
    async function redirectToStep() {
      try {
        // Check if onboarding is enabled (also gets chat flag for redirect)
        const { shouldShowOnboarding } = await getOnboardingStatus();

        if (!shouldShowOnboarding) {
          router.replace("/");
          return;
        }

        const onboarding = await resolveResponse(getV1OnboardingState());

        // Handle completed onboarding
        if (onboarding.completedSteps.includes("GET_RESULTS")) {
          router.replace("/");
          return;
        }

        // Redirect to appropriate step based on completed steps
        if (onboarding.completedSteps.includes("AGENT_INPUT")) {
          router.push("/onboarding/5-run");
          return;
        }

        if (onboarding.completedSteps.includes("AGENT_NEW_RUN")) {
          router.push("/onboarding/5-run");
          return;
        }

        if (onboarding.completedSteps.includes("AGENT_CHOICE")) {
          router.push("/onboarding/5-run");
          return;
        }

        if (onboarding.completedSteps.includes("INTEGRATIONS")) {
          router.push("/onboarding/4-agent");
          return;
        }

        if (onboarding.completedSteps.includes("USAGE_REASON")) {
          router.push("/onboarding/3-services");
          return;
        }

        if (onboarding.completedSteps.includes("WELCOME")) {
          router.push("/onboarding/2-reason");
          return;
        }

        // Default: redirect to first step
        router.push("/onboarding/1-welcome");
      } catch (error) {
        console.error("Failed to determine onboarding step:", error);
        router.replace("/");
      }
    }

    redirectToStep();
  }, [router]);

  return <LoadingSpinner size="large" cover />;
}
