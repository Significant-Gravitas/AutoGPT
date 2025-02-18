"use client";
import { UserOnboarding } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";

const OnboardingContext = createContext<
  | {
      state: UserOnboarding | null;
      setState: (state: Partial<UserOnboarding>) => void;
    }
  | undefined
>(undefined);

export function useOnboarding(step?: number) {
  const context = useContext(OnboardingContext);
  if (!context)
    throw new Error("useOnboarding must be used within OnboardingLayout");

  useEffect(() => {
    if (!step) return;

    context.setState({ step });
  }, [step]);

  return context;
}

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  const [state, setStateRaw] = useState<UserOnboarding | null>(null);
  const api = useBackendAPI();

  useEffect(() => {
    const fetchOnboarding = async () => {
      const onboarding = await api.getUserOnboarding();
      setStateRaw(onboarding);
      console.log("userOnboarding", onboarding);
    };
    fetchOnboarding();
  }, [api]);

  const setState = (newState: Partial<UserOnboarding>) => {
    setStateRaw((prev) => {
      if (!prev) {
        // Handle initial state
        return {
          step: 1,
          integrations: [],
          isCompleted: false,
          ...newState,
        } as UserOnboarding;
      }
      return { ...prev, ...newState };
    });
  };

  return (
    <OnboardingContext.Provider value={{ state, setState }}>
      <div className="flex min-h-screen w-full items-center justify-center bg-gray-100">
        <div className="mx-auto flex w-full flex-col items-center">
          {children}
        </div>
      </div>
    </OnboardingContext.Provider>
  );
}
