"use client";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import Link from "next/link";

const OnboardingContext = createContext<
  | {
      state: UserOnboarding | null;
      updateState: (
        state: Omit<Partial<UserOnboarding>, "rewardedFor">,
      ) => void;
      step: number;
      setStep: (step: number) => void;
      completeStep: (step: OnboardingStep) => void;
      incrementRuns: () => void;
    }
  | undefined
>(undefined);

export function useOnboarding(step?: number, completeStep?: OnboardingStep) {
  const context = useContext(OnboardingContext);
  if (!context)
    throw new Error("useOnboarding must be used within an OnboardingProvider");

  useEffect(() => {
    if (
      !completeStep ||
      !context.state ||
      context.state.completedSteps.includes(completeStep)
    )
      return;

    context.updateState({
      completedSteps: [...context.state.completedSteps, completeStep],
    });
  }, [completeStep, context, context.updateState]);

  useEffect(() => {
    if (step && context.step !== step) {
      context.setStep(step);
    }
  }, [step, context]);

  return context;
}

export default function OnboardingProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [state, setState] = useState<UserOnboarding | null>(null);
  // Step is used to control the progress bar, it's frontend only
  const [step, setStep] = useState(1);
  const [npsDialogOpen, setNpsDialogOpen] = useState(false);
  const api = useBackendAPI();
  const pathname = usePathname();
  const router = useRouter();
  const { user, isUserLoading } = useSupabase();

  useEffect(() => {
    const fetchOnboarding = async () => {
      const enabled = await api.isOnboardingEnabled();
      if (!enabled && pathname.startsWith("/onboarding")) {
        router.push("/marketplace");
        return;
      }
      const onboarding = await api.getUserOnboarding();
      setState((prev) => ({ ...onboarding, ...prev }));

      // Redirect outside onboarding if completed
      // If user did CONGRATS step, that means they completed introductory onboarding
      if (
        onboarding.completedSteps.includes("CONGRATS") &&
        pathname.startsWith("/onboarding") &&
        !pathname.startsWith("/onboarding/reset")
      ) {
        router.push("/marketplace");
      }
    };
    if (isUserLoading || !user) {
      return;
    }
    fetchOnboarding();
  }, [api, pathname, router, user, isUserLoading]);

  const updateState = useCallback(
    (newState: Omit<Partial<UserOnboarding>, "rewardedFor">) => {
      setState((prev) => {
        api.updateUserOnboarding(newState);

        if (!prev) {
          // Handle initial state
          return {
            completedSteps: [],
            notificationDot: false,
            notified: [],
            rewardedFor: [],
            usageReason: null,
            integrations: [],
            otherIntegrations: null,
            selectedStoreListingVersionId: null,
            agentInput: null,
            onboardingAgentExecutionId: null,
            agentRuns: 0,
            ...newState,
          };
        }
        return { ...prev, ...newState };
      });
    },
    [api, setState],
  );

  const completeStep = useCallback(
    (step: OnboardingStep) => {
      if (!state || state.completedSteps.includes(step)) return;

      updateState({
        completedSteps: [...state.completedSteps, step],
      });
    },
    [state, updateState],
  );

  const incrementRuns = useCallback(() => {
    if (!state || state.completedSteps.includes("RUN_AGENTS")) return;

    const finished = state.agentRuns + 1 >= 10;
    setNpsDialogOpen(finished);
    updateState({
      agentRuns: state.agentRuns + 1,
      ...(finished && {
        completedSteps: [...state.completedSteps, "RUN_AGENTS"],
      }),
    });
  }, [api, state]);

  return (
    <OnboardingContext.Provider
      value={{ state, updateState, step, setStep, completeStep, incrementRuns }}
    >
      <Dialog onOpenChange={setNpsDialogOpen} open={npsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>We&apos;d love your feedback</DialogTitle>
            <DialogDescription>
              You&apos;ve run 10 agents — amazing! We&apos;re constantly
              improving the platform, and your thoughts help shape what we build
              next. This 1-minute form is just a few quick questions to share
              how things are going.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="justify-end">
            <Button
              type="button"
              variant="outline"
              onClick={() => setNpsDialogOpen(false)}
            >
              Cancel
            </Button>
            <Link href="https://tally.so/r/w4El0b" target="_blank">
              <Button type="button" onClick={() => setNpsDialogOpen(false)}>
                Give Feedback
              </Button>
            </Link>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {children}
    </OnboardingContext.Provider>
  );
}
