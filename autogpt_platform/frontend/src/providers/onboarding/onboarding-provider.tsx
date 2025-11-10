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
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useOnboardingTimezoneDetection } from "@/hooks/useOnboardingTimezoneDetection";
import { OnboardingStep, UserOnboarding } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  calculateConsecutiveDays,
  updateOnboardingState,
  getRunMilestoneSteps,
  processOnboardingData,
  shouldRedirectFromOnboarding,
} from "./helpers";
import { resolveResponse } from "@/app/api/helpers";
import { getV1OnboardingState, patchV1UpdateOnboardingState, postV1CompleteOnboardingStep } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { UserOnboardingUpdate } from "@/app/api/__generated__/models/userOnboardingUpdate";
import { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";

type FrontendOnboardingStep = PostV1CompleteOnboardingStepStep;

const OnboardingContext = createContext<
  | {
      state: UserOnboarding | null;
      updateState: (
        state: Omit<Partial<UserOnboarding>, "rewardedFor">,
      ) => void;
      step: number;
      setStep: (step: number) => void;
      completeStep: (step: FrontendOnboardingStep) => void;
      incrementRuns: () => void;
      fetchOnboarding: () => Promise<UserOnboarding>;
    }
  | undefined
>(undefined);

export function useOnboarding(step?: number, completeStep?: FrontendOnboardingStep) {
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
  const [step, setStep] = useState(1);
  const [npsDialogOpen, setNpsDialogOpen] = useState(false);
  const hasInitialized = useRef(false);
  const isMounted = useRef(true);
  const pendingUpdatesRef = useRef<Set<Promise<void>>>(new Set());
  const { toast } = useToast();

  const api = useBackendAPI();
  const pathname = usePathname();
  const router = useRouter();
  const { user, isUserLoading } = useSupabase();

  useOnboardingTimezoneDetection();

  // Cleanup effect to track mount state and cancel pending operations
  useEffect(() => {
    isMounted.current = true;

    return () => {
      isMounted.current = false;

      // Wait for pending updates to complete before unmounting
      pendingUpdatesRef.current.forEach((promise) => {
        promise.catch(() => {});
      });

      pendingUpdatesRef.current.clear();
    };
  }, []);

  const isOnOnboardingRoute = pathname.startsWith("/onboarding");

  const fetchOnboarding = useCallback(async () => {
    const onboarding = await resolveResponse(getV1OnboardingState());
    const processedOnboarding = processOnboardingData(onboarding);
    setState(processedOnboarding);
    return processedOnboarding;
  }, []);

  useEffect(() => {
    // Prevent multiple initializations
    if (hasInitialized.current || isUserLoading || !user) {
      return;
    }

    hasInitialized.current = true;

    async function initializeOnboarding() {
      try {
        // Check onboarding enabled only for onboarding routes
        if (isOnOnboardingRoute) {
          const enabled = await api.isOnboardingEnabled();
          if (!enabled) {
            router.push("/marketplace");
            return;
          }
        }

        const onboarding = await fetchOnboarding();

        // Handle redirects for completed onboarding
        if (
          isOnOnboardingRoute &&
          shouldRedirectFromOnboarding(
            onboarding.completedSteps,
            pathname,
          )
        ) {
          router.push("/marketplace");
        }
      } catch (error) {
        console.error("Failed to initialize onboarding:", error);

        toast({
          title: "Failed to initialize onboarding",
          variant: "destructive",
        });

        hasInitialized.current = false; // Allow retry on next render
      }
    }

    initializeOnboarding();
  }, [api, isOnOnboardingRoute, router, user, isUserLoading, pathname, fetchOnboarding, toast]);

  const updateState = useCallback(
    (newState: UserOnboardingUpdate) => {
      // Update local state immediately
      setState((prev) => updateOnboardingState(prev, newState));

      const updatePromise = (async () => {
        try {
          if (!isMounted.current) return;
          await patchV1UpdateOnboardingState(newState);
        } catch (error) {
          if (isMounted.current) {
            console.error("Failed to update user onboarding:", error);
          }

          toast({
            title: "Failed to update user onboarding",
            variant: "destructive",
          });
        }
      })();

      // Track this pending update
      pendingUpdatesRef.current.add(updatePromise);

      updatePromise.finally(() => {
        pendingUpdatesRef.current.delete(updatePromise);
      });
    },
    [api],
  );

  const completeStep = useCallback(
    (step: FrontendOnboardingStep) => {
      if (!state?.completedSteps?.includes(step)) {
        postV1CompleteOnboardingStep({ step });
        fetchOnboarding();
      }
    },
    [state?.completedSteps, updateState, fetchOnboarding],
  );

  const incrementRuns = useCallback(() => {
    if (!state?.completedSteps) return;

    const newRunCount = state.agentRuns + 1;
    const consecutiveData = calculateConsecutiveDays(
      state.lastRunAt,
      state.consecutiveRunDays,
    );

    const milestoneSteps = getRunMilestoneSteps(
      newRunCount,
      consecutiveData.consecutiveRunDays,
    );

    // Show NPS dialog at 10 runs
    if (newRunCount === 10) {
      setNpsDialogOpen(true);
    }

    // updateState({
    //   agentRuns: newRunCount,
    //   completedSteps: Array.from(
    //     new Set([...state.completedSteps, ...milestoneSteps]),
    //   ),
    //   ...consecutiveData,
    // });
  }, [state, updateState]);

  return (
    <OnboardingContext.Provider
      value={{ state, updateState, step, setStep, completeStep, incrementRuns, fetchOnboarding }}
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
