import { postV1ResetOnboardingProgress } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { redirect } from "next/navigation";

export default async function OnboardingResetPage() {
  await postV1ResetOnboardingProgress();
  redirect("/onboarding/1-welcome");
}
