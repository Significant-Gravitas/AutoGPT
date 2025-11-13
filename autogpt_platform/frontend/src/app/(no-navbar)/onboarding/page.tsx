import { redirect } from "next/navigation";
import { finishOnboarding } from "./6-congrats/actions";
import { resolveResponse, shouldShowOnboarding } from "@/app/api/helpers";
import { getV1OnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";

// Force dynamic rendering to avoid static generation issues with cookies
export const dynamic = "force-dynamic";

export default async function OnboardingPage() {
  const isOnboardingEnabled = await shouldShowOnboarding();

  if (!isOnboardingEnabled) {
    redirect("/marketplace");
  }

  const onboarding = await resolveResponse(getV1OnboardingState());

  // CONGRATS is the last step in intro onboarding
  if (onboarding.completedSteps.includes("GET_RESULTS"))
    redirect("/marketplace");
  else if (onboarding.completedSteps.includes("CONGRATS")) finishOnboarding();
  else if (onboarding.completedSteps.includes("AGENT_INPUT"))
    redirect("/onboarding/5-run");
  else if (onboarding.completedSteps.includes("AGENT_NEW_RUN"))
    redirect("/onboarding/5-run");
  else if (onboarding.completedSteps.includes("AGENT_CHOICE"))
    redirect("/onboarding/5-run");
  else if (onboarding.completedSteps.includes("INTEGRATIONS"))
    redirect("/onboarding/4-agent");
  else if (onboarding.completedSteps.includes("USAGE_REASON"))
    redirect("/onboarding/3-services");
  else if (onboarding.completedSteps.includes("WELCOME"))
    redirect("/onboarding/2-reason");

  redirect("/onboarding/1-welcome");
}
