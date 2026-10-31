import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboardingWizardStore } from "../../store";
import { ACCESS_PROVIDERS } from "./accessProviders";
import { useSignInProvider } from "./useSignInProvider";

export function useAccessStep() {
  const { isLoggedIn } = useAuth();
  const { provider } = useSignInProvider(isLoggedIn);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  return {
    access: provider ? ACCESS_PROVIDERS[provider] : null,
    handleContinue: nextStep,
  };
}
