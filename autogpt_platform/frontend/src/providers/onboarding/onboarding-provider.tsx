"use client";
import {
  getV1CheckIfOnboardingIsCompleted,
  getV1OnboardingState,
  patchV1UpdateOnboardingState,
  postV1CompleteOnboardingStep,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";
import { resolveResponse } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useOnboardingTimezoneDetection } from "@/hooks/useOnboardingTimezoneDetection";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import {
  ApiError,
  UserOnboarding,
  WebSocketNotification,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
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
  decideOnboardingRedirect,
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
  const hasInitialized = useRef(false);
  const isMounted = useRef(true);
  const pendingUpdatesRef = useRef<Set<Promise<void>>>(new Set());
  const { toast } = useToast();

  const api = useBackendAPI();
  const pathname = usePathname();
  const searchParams = useSearchParams();
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
  // Logged-in users sitting on the auth pages need to be routed onward by us;
  // otherwise the signup/login pages show their `isLoggedIn` loader forever.
  // Handling them here (instead of in useSignupPage/useLoginPage) avoids the
  // /signup → / → /copilot → /onboarding bounce that flashes /copilot.
  const isOnAuthRoute = pathname === "/signup" || pathname === "/login";
  // When `/login?next=/profile` is hit, useLoginPage/useSignupPage fires its
  // own redirect to the requested target. Skip our redirect in that window so
  // the deep link isn't clobbered by the awaited completion check. Sanitize
  // with the same helper the auth pages use — otherwise an unsafe value (e.g.
  // `?next=https://evil.site`) would defer us here while the auth page drops
  // it as `null`, deadlocking the user on the auth loader.
  const hasPendingAuthDeepLink =
    isOnAuthRoute && sanitizeAuthNext(searchParams.get("next")) !== null;

  const fetchOnboarding = useCallback(async () => {
    const onboarding = await resolveResponse(getV1OnboardingState());
    const processedOnboarding = fromBackendUserOnboarding(onboarding);
    if (isMounted.current) {
      setState(processedOnboarding);
    }
    return processedOnboarding;
  }, []);

  // If a logged-in user navigates back to /signup or /login after the
  // initialize-once effect already ran (e.g., browser back from /onboarding,
  // manual URL edit), the guard below would early-return and leave them
  // stranded on the auth page's `isLoggedIn` loader. Reset the guard on
  // re-entry so the main effect re-runs and routes them away.
  useEffect(() => {
    if (isLoggedIn && isOnAuthRoute) {
      hasInitialized.current = false;
    }
  }, [isLoggedIn, isOnAuthRoute]);

  useEffect(() => {
    // Prevent multiple initializations
    if (hasInitialized.current || !isLoggedIn) {
      return;
    }

    // Defer to the auth-page's own deep-link redirect (`?next=…`) instead of
    // racing it. Don't mark hasInitialized so the next pathname change
    // (after the auth page redirects) gets us properly initialized.
    if (hasPendingAuthDeepLink) {
      return;
    }

    hasInitialized.current = true;

    async function initializeOnboarding() {
      try {
        const { is_completed } = await resolveResponse(
          getV1CheckIfOnboardingIsCompleted(),
        );

        const redirectTarget = decideOnboardingRedirect({
          isCompleted: is_completed,
          isOnOnboardingRoute,
          isOnAuthRoute,
          hasPendingAuthDeepLink,
        });
        if (redirectTarget) {
          router.replace(redirectTarget);
          return;
        }

        const onboarding = await fetchOnboarding();

        // Handle redirects for completed onboarding
        if (
          isOnOnboardingRoute &&
          shouldRedirectFromOnboarding(onboarding.completedSteps, pathname)
        ) {
          router.replace("/copilot");
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          hasInitialized.current = false;
          return;
        }

        console.error("Failed to initialize onboarding:", error);

        toast({
          title: "Failed to initialize onboarding",
          variant: "destructive",
        });

        hasInitialized.current = false; // Allow retry on next render
      }
    }

    initializeOnboarding();
  }, [
    api,
    isOnOnboardingRoute,
    isOnAuthRoute,
    hasPendingAuthDeepLink,
    router,
    isLoggedIn,
    pathname,
  ]);

  const handleOnboardingNotification = useCallback(
    (notification: WebSocketNotification) => {
      if (!isLoggedIn || notification.type !== "onboarding") {
        return;
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
      {children}
    </OnboardingContext.Provider>
  );
}
