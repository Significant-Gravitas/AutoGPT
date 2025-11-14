"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";

export default function OnboardingPage() {
  const router = useRouter();
  const api = useBackendAPI();

  useEffect(() => {
    async function redirectToStep() {
      try {
        // Check if onboarding is enabled
        const isEnabled = await api.isOnboardingEnabled();
        if (!isEnabled) {
          router.push("/");
          return;
        }

        const onboarding = await api.getUserOnboarding();

        // Handle completed onboarding
        if (onboarding.completedSteps.includes("GET_RESULTS")) {
          router.push("/");
          return;
        }

        // Handle CONGRATS - add agent to library and redirect
        if (onboarding.completedSteps.includes("CONGRATS")) {
          if (onboarding.selectedStoreListingVersionId) {
            try {
              const libraryAgent = await api.addMarketplaceAgentToLibrary(
                onboarding.selectedStoreListingVersionId,
              );
              router.push(`/library/agents/${libraryAgent.id}`);
            } catch (error) {
              console.error("Failed to add agent to library:", error);
              router.push("/library");
            }
          } else {
            router.push("/library");
          }
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
        router.push("/");
      }
    }

    redirectToStep();
  }, [api, router]);

  return <LoadingSpinner size="large" cover />;
}
