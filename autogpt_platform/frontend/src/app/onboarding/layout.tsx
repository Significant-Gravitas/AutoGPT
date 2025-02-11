'use client';
import { useRouter } from 'next/navigation';
import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

type OnboardingState = {
  step: number;
  usageReason?: string;
  integrations?: string[];
  chosenAgentId?: string;
  agentInput?: { [key: string]: string };
};

const OnboardingContext = createContext<{
  state: OnboardingState;
  setState: (state: Partial<OnboardingState>) => void;
} | undefined>(undefined);

export function useOnboarding(step: number) {
  const context = useContext(OnboardingContext);
  if (!context) throw new Error('useOnboarding must be used within OnboardingLayout');

  useEffect(() => {
    if (step > context.state.step) {
      context.setState({ step });
    }
  }, [step]);

  return context;
}

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  const [state, setStateRaw] = useState<OnboardingState>({ step: 0 });

  //todo kcze user

  const setState = (newState: Partial<OnboardingState>) => {
    // Don't update step if it's lower than current
    if (newState.step && newState.step < state.step) {
      delete newState.step;
    }
    setStateRaw(prev => ({ ...prev, ...newState }));
  };

  return (
    <OnboardingContext.Provider value={{ state, setState }}>
      <div className="min-h-screen w-full flex items-center justify-center bg-gray-100">
        <div className="max-w-2xl w-full mx-auto px-4 flex flex-col items-center">
          {children}
        </div>
      </div>
    </OnboardingContext.Provider>
  );
}
