"use client";
import { Button } from "@/components/__legacy__/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { OnboardingStep, UserOnboarding } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useOnboardingTimezoneDetection } from "@/hooks/useOnboardingTimezoneDetection";
import Link from "next/link";
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
      !context.state.completedSteps ||
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

  // Automatically detect and set timezone for new users during onboarding
  useOnboardingTimezoneDetection();

  const isOnOnboardingRoute = pathname.startsWith("/onboarding");

  useEffect(() => {
    // Only run heavy onboarding API calls if user is logged in and not loading
    if (isUserLoading || !user) {
      return;
    }

    const fetchOnboarding = async () => {
      try {
        // For non-onboarding routes, we still need basic onboarding state for step completion
        // but we can skip the expensive isOnboardingEnabled() check
        if (isOnOnboardingRoute) {
          // Only check if onboarding is enabled when user is actually on onboarding routes
          const enabled = await api.isOnboardingEnabled();
          if (!enabled) {
            router.push("/marketplace");
            return;
          }
        }

        // Always fetch user onboarding state for step completion functionality
        const onboarding = await api.getUserOnboarding();

        // Only update state if onboarding data is valid
        if (onboarding) {
          //todo kcze this is a patch because only TRIGGER_WEBHOOK is set on the backend and then overwritten by the frontend
          const completeWebhook =
            onboarding.rewardedFor.includes("TRIGGER_WEBHOOK") &&
            !onboarding.completedSteps.includes("TRIGGER_WEBHOOK")
              ? (["TRIGGER_WEBHOOK"] as OnboardingStep[])
              : [];

          setState((prev) => ({
            ...onboarding,
            completedSteps: [...completeWebhook, ...onboarding.completedSteps],
            lastRunAt: new Date(onboarding.lastRunAt || ""),
            ...prev,
          }));

          // Only handle onboarding redirects when user is on onboarding routes
          if (isOnOnboardingRoute) {
            // Redirect outside onboarding if completed
            // If user did CONGRATS step, that means they completed introductory onboarding
            if (
              onboarding.completedSteps &&
              onboarding.completedSteps.includes("CONGRATS") &&
              !pathname.startsWith("/onboarding/reset")
            ) {
              router.push("/marketplace");
            }
          }
        }
      } catch (error) {
        console.error("Failed to fetch onboarding data:", error);
        // Don't update state on error to prevent null access issues
      }
    };

    fetchOnboarding();
  }, [api, isOnOnboardingRoute, router, user, isUserLoading]);

  const updateState = useCallback(
    (newState: Omit<Partial<UserOnboarding>, "rewardedFor">) => {
      setState((prev) => {
        if (!prev) {
          // Handle initial state
          return {
            completedSteps: [],
            walletShown: true,
            notified: [],
            rewardedFor: [],
            usageReason: null,
            integrations: [],
            otherIntegrations: null,
            selectedStoreListingVersionId: null,
            agentInput: null,
            onboardingAgentExecutionId: null,
            agentRuns: 0,
            lastRunAt: null,
            consecutiveRunDays: 0,
            ...newState,
          };
        }
        return { ...prev, ...newState };
      });
      // Make the API call asynchronously to not block render
      setTimeout(() => {
        api.updateUserOnboarding(newState).catch((error) => {
          console.error("Failed to update user onboarding:", error);
        });
      }, 0);
    },
    [api],
  );

  const completeStep = useCallback(
    (step: OnboardingStep) => {
      if (
        !state ||
        !state.completedSteps ||
        state.completedSteps.includes(step)
      )
        return;

      updateState({
        completedSteps: [...state.completedSteps, step],
      });
    },
    [state, updateState],
  );

  const isToday = useCallback((date: Date) => {
    const today = new Date();

    return (
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear()
    );
  }, []);

  const isYesterday = useCallback((date: Date): boolean => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);

    return (
      date.getDate() === yesterday.getDate() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getFullYear() === yesterday.getFullYear()
    );
  }, []);

  const incrementRuns = useCallback(() => {
    if (!state || !state.completedSteps) return;

    const tenRuns = state.agentRuns + 1 === 10;
    const hundredRuns = state.agentRuns + 1 === 100;
    // Calculate if it's a run on a consecutive day
    // If the last run was yesterday, increment days
    // Otherwise, if the last run was *not* today reset it (already checked that it wasn't yesterday at this point)
    // Otherwise, don't do anything (the last run was today)
    const consecutive =
      state.lastRunAt === null || isYesterday(state.lastRunAt)
        ? {
            lastRunAt: new Date(),
            consecutiveRunDays: state.consecutiveRunDays + 1,
          }
        : !isToday(state.lastRunAt)
          ? { lastRunAt: new Date(), consecutiveRunDays: 1 }
          : {};

    setNpsDialogOpen(tenRuns);
    updateState({
      agentRuns: state.agentRuns + 1,
      completedSteps: [
        ...state.completedSteps,
        ...(tenRuns ? (["RUN_AGENTS"] as OnboardingStep[]) : []),
        ...(hundredRuns ? (["RUN_AGENTS_100"] as OnboardingStep[]) : []),
        ...(consecutive.consecutiveRunDays === 3
          ? (["RUN_3_DAYS"] as OnboardingStep[])
          : []),
        ...(consecutive.consecutiveRunDays === 14
          ? (["RUN_14_DAYS"] as OnboardingStep[])
          : []),
      ],
      ...consecutive,
    });
  }, [state, updateState]);

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
