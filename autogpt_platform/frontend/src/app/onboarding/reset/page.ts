import BackendAPI from "@/lib/autogpt-server-api";
import { redirect } from "next/navigation";

export default async function OnboardingResetPage() {
  const api = new BackendAPI();
  await api.updateUserOnboarding({
    completedSteps: [],
    usageReason: null,
    integrations: [],
    otherIntegrations: "",
    selectedStoreListingVersionId: null,
    agentInput: {},
  });
  redirect("/onboarding/1-welcome");
}
