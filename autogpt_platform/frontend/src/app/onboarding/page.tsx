import BackendAPI from "@/lib/autogpt-server-api";
import { redirect } from "next/navigation";

export default async function OnboardingPage() {
  const api = new BackendAPI();

  if (!api.isOnboardingEnabled()) {
    redirect("/marketplace");
  }

  const onboarding = await api.getUserOnboarding();

  // CONGRATS is the last step in intro onboarding
  if (onboarding.completedSteps.includes("CONGRATS")) redirect("/marketplace");
  else if (onboarding.completedSteps.includes("AGENT_INPUT"))
    redirect("/onboarding/6-congrats");
  else if (onboarding.completedSteps.includes("AGENT_NEW_RUN"))
    redirect("/onboarding/5-run");
  else if (onboarding.completedSteps.includes("AGENT_CHOICE"))
    redirect("/onboarding/5-agent");
  else if (onboarding.completedSteps.includes("INTEGRATIONS"))
    redirect("/onboarding/4-agent");
  else if (onboarding.completedSteps.includes("USAGE_REASON"))
    redirect("/onboarding/3-services");
  else if (onboarding.completedSteps.includes("WELCOME"))
    redirect("/onboarding/2-reason");

  redirect("/onboarding/1-welcome");
}
