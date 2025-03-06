import BackendAPI from "@/lib/autogpt-server-api";
import { redirect } from "next/navigation";

export default async function OnboardingPage() {
  const api = new BackendAPI();
  const onboarding = await api.getUserOnboarding();

  switch (onboarding.step) {
    case 1:
      redirect("/onboarding/1-welcome");
    case 2:
      redirect("/onboarding/2-reason");
    case 3:
      redirect("/onboarding/3-services");
    case 4:
      redirect("/onboarding/4-agent");
    case 5:
      redirect("/onboarding/5-run");
    default:
      redirect("/onboarding/1-welcome");
  }
}
