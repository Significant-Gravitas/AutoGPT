"use client";
import { OnboardingStep, UserOnboarding } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

const OnboardingContext = createContext<
  | {
    state: UserOnboarding | null;
    updateState: (state: Partial<UserOnboarding>) => void;
  }
  | undefined
>(undefined);

export function useOnboarding(completeStep?: OnboardingStep) {
  const context = useContext(OnboardingContext);
  if (!context)
    throw new Error("useOnboarding must be used within /onboarding pages");

  useEffect(() => {
    if (!completeStep || !context.state || context.state.completedSteps.includes(completeStep)) return;

    context.updateState({ completedSteps: [...context.state.completedSteps, completeStep] });
  }, [completeStep, context.state, context.updateState]);

  return context;
}

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  const [state, setState] = useState<UserOnboarding | null>(null);
  const api = useBackendAPI();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    const fetchOnboarding = async () => {
      const enabled = await api.isOnboardingEnabled();
      if (!enabled) {
        router.push("/library");
        return;
      }
      const onboarding = await api.getUserOnboarding();
      setState(onboarding);

      // Redirect outside onboarding if completed
      // If user did CONGRATS step, that means they completed introductory onboarding
      if (onboarding.completedSteps.includes("CONGRATS") && !pathname.startsWith("/onboarding/reset")) {
        router.push("/library");
      }
    };
    fetchOnboarding();
  }, [api, pathname, router]);

  const updateState = useCallback(
    (newState: Partial<UserOnboarding>) => {
      const sendState = (state: Partial<UserOnboarding>) => {
        if (!state) return;

        api.updateUserOnboarding(state);
      };

      setState((prev) => {
        // We want to send updates only when completedSteps is updated
        // to avoid api calls on every small change
        if (newState.completedSteps) {
          sendState({ ...prev, ...newState });
        }

        if (!prev) {
          // Handle initial state
          return {
            completedSteps: [],
            usageReason: null,
            integrations: [],
            otherIntegrations: null,
            selectedAgentCreator: null,
            selectedAgentSlug: null,
            agentInput: null,
            ...newState,
          };
        }
        return { ...prev, ...newState };
      });
    },
    [api, setState],
  );

  return (
    <OnboardingContext.Provider value={{ state, updateState }}>
      <div className="flex min-h-screen w-full items-center justify-center bg-gray-100">
        <div className="mx-auto flex w-full flex-col items-center">
          {children}
        </div>
      </div>
    </OnboardingContext.Provider>
  );
}
