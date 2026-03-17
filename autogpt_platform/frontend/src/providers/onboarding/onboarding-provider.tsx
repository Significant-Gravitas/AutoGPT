"use client";
import {
  getV1IsOnboardingEnabled,
  getV1OnboardingState,
  patchV1UpdateOnboardingState,
  postV1CompleteOnboardingStep,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";
import { resolveResponse } from "@/app/api/helpers";
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
import {
  UserOnboarding,
  WebSocketNotification,
} from "@/lib/autogpt-server-api";
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
  fromBackendUserOnboarding,
  LocalOnboardingStateUpdate,
  shouldRedirectFromOnboarding,
  updateOnboardingState,
} from "./helpers";

type FrontendOnboardingStep = PostV1CompleteOnboardingStepStep;

const OnboardingContext = createContext<
  | {
      state: UserOnboarding | null;
      updateState: (state: LocalOnboardingStateUpdate) => void;
      step: number;
      setStep: (step: number) => void;
      completeStep: (step: FrontendOnboardingStep) => void;
    }
  | undefined
>(undefined);

export function useOnboarding(
  step?: number,
  completeStep?: FrontendOnboardingStep,
) {
  const context = useContext(OnboardingContext);

  if (!context)
    throw new Error("useOnboarding must be used within an OnboardingProvider");

  useEffect(() => {
    if (
      !completeStep ||
      !context.state ||
      context.state.completedSteps.includes(completeStep)
    ) {
      return;
    }

    context.completeStep(completeStep);
  }, [completeStep, context]);

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
  const { isLoggedIn } = useSupabase();

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
    const processedOnboarding = fromBackendUserOnboarding(onboarding);
    if (isMounted.current) {
      setState(processedOnboarding);
    }
    return processedOnboarding;
  }, []);

  useEffect(() => {
    // Prevent multiple initializations
    if (hasInitialized.current || !isLoggedIn) {
      return;
    }

    hasInitialized.current = true;

    async function initializeOnboarding() {
      try {
        // Check onboarding enabled only for onboarding routes
        if (isOnOnboardingRoute) {
          const enabled = await resolveResponse(getV1IsOnboardingEnabled());
          if (!enabled) {
            router.push("/");
            return;
          }
        }

        const onboarding = await fetchOnboarding();

        // Handle redirects for completed onboarding
        if (
          isOnOnboardingRoute &&
          shouldRedirectFromOnboarding(onboarding.completedSteps, pathname)
        ) {
          router.push("/");
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
  }, [api, isOnOnboardingRoute, router, isLoggedIn, pathname]);

  const handleOnboardingNotification = useCallback(
    (notification: WebSocketNotification) => {
      if (!isLoggedIn || notification.type !== "onboarding") {
        return;
      }

      if (notification.step === "RUN_AGENTS") {
        setNpsDialogOpen(true);
      }

      fetchOnboarding().catch((error) => {
        console.error(
          "Failed to refresh onboarding after notification:",
          error,
        );
      });
    },
    [fetchOnboarding, isLoggedIn],
  );

  useEffect(() => {
    const detachMessage = api.onWebSocketMessage(
      "notification",
      handleOnboardingNotification,
    );

    if (isLoggedIn) {
      api.connectWebSocket();
    }

    return () => {
      detachMessage();
    };
  }, [api, handleOnboardingNotification, isLoggedIn]);

  const updateState = useCallback(
    (newState: LocalOnboardingStateUpdate) => {
      if (!isLoggedIn) {
        return;
      }

      setState((prev) => updateOnboardingState(prev, newState));

      const updatePromise = (async () => {
        try {
          if (!isMounted.current) return;
          await patchV1UpdateOnboardingState(newState);
        } catch (error) {
          console.error("Failed to update user onboarding:", error);

          toast({
            title: "Failed to update user onboarding",
            variant: "destructive",
          });
        }
      })();

      pendingUpdatesRef.current.add(updatePromise);

      updatePromise.finally(() => {
        pendingUpdatesRef.current.delete(updatePromise);
      });
    },
    [toast, isLoggedIn, fetchOnboarding, api, setState],
  );

  const completeStep = useCallback(
    (step: FrontendOnboardingStep) => {
      if (!isLoggedIn || state?.completedSteps?.includes(step)) {
        return;
      }

      const completionPromise = (async () => {
        try {
          await postV1CompleteOnboardingStep({ step });
          await fetchOnboarding();
        } catch (error) {
          if (isMounted.current) {
            console.error("Failed to complete onboarding step:", error);
          }

          toast({
            title: "Failed to complete onboarding step",
            variant: "destructive",
          });
        }
      })();

      pendingUpdatesRef.current.add(completionPromise);
      completionPromise.finally(() => {
        pendingUpdatesRef.current.delete(completionPromise);
      });
    },
    [isLoggedIn, state?.completedSteps, fetchOnboarding, toast],
  );

  return (
    <OnboardingContext.Provider
      value={{
        state,
        updateState,
        step,
        setStep,
        completeStep,
      }}
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
