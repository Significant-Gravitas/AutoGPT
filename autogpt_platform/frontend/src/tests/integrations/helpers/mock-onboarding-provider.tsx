import { createContext, useContext, ReactNode } from "react";
import { UserOnboarding } from "@/lib/autogpt-server-api";
import { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";
import type { LocalOnboardingStateUpdate } from "@/providers/onboarding/helpers";

const MockOnboardingContext = createContext<{
  state: UserOnboarding | null;
  updateState: (state: LocalOnboardingStateUpdate) => void;
  step: number;
  setStep: (step: number) => void;
  completeStep: (step: PostV1CompleteOnboardingStepStep) => void;
}>({
  state: null,
  updateState: () => {},
  step: 1,
  setStep: () => {},
  completeStep: () => {},
});

export function useOnboarding(
  _step?: number,
  _completeStep?: PostV1CompleteOnboardingStepStep,
) {
  const context = useContext(MockOnboardingContext);
  return context;
}

interface Props {
  children: ReactNode;
}

export function MockOnboardingProvider({ children }: Props) {
  return (
    <MockOnboardingContext.Provider
      value={{
        state: null,
        updateState: () => {},
        step: 1,
        setStep: () => {},
        completeStep: () => {},
      }}
    >
      {children}
    </MockOnboardingContext.Provider>
  );
}
