import BackendAPI from "@/lib/autogpt-server-api";
import { redirect } from "next/navigation";

export default async function OnboardingResetPage() {
  const api = new BackendAPI();
  await api.updateUserOnboarding({
    step: 1,
    usageReason: undefined,
    integrations: [],
    otherIntegrations: undefined,
    selectedAgentCreator: undefined,
    selectedAgentSlug: undefined,
    agentInput: undefined,
    isCompleted: false,
  });
  redirect("/onboarding/1-welcome");
}
