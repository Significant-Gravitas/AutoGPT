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
import { getAuthDeepLink, withAuthTracking } from "@/lib/auth-redirect";
import {
  ApiError,
  UserOnboarding,
  WebSocketNotification,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { hasImpersonation } from "@/lib/impersonation";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import {
  isAbortError,
  isNetworkError,
  retryTransient,
} from "@/lib/transient-errors";
import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  decideOnboardingRedirect,
  fromBackendUserOnboarding,
  getOnboardingRoute,
  hasCompletedOnboarding,
  LocalOnboardingStateUpdate,
  shouldRedirectFromOnboarding,
  updateOnboardingState,
  withCompletion,
} from "./helpers";
import { useOnboardingCompletion } from "./useOnboardingCompletion";
import { AuthDeepLinkWatcher } from "./AuthDeepLinkWatcher";
import isEqual from "lodash/isEqual";

type FrontendOnboardingStep = PostV1CompleteOnboardingStepStep;

const OnboardingContext = createContext<
  | {
      state: UserOnboarding | null;
      // Whether the signed-in user has finished onboarding; `null` until that
      // is known. Prompts that would send a new user somewhere else (the
      // paywall, the out-of-credits dialog) wait for `true`.
      isOnboardingComplete: boolean | null;
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

// When `/login?next=/profile` is hit, useLoginPage/useSignupPage fires its own
// redirect to the requested target, so the provider must not race it. Sanitize
// with the same helper the auth pages use — otherwise an unsafe value (e.g.
// `?next=https://evil.site`) or a default landing (`?next=/home`) would defer
// us here while the auth page drops it as `null`, deadlocking the user on the
// auth loader. Read from the location (only ever inside an effect, after Next
// has committed the URL) instead of useSearchParams, which would re-render
// this app-root provider on every search-param write anywhere in the app.
// AuthDeepLinkWatcher re-runs the effect when `next` changes on an auth route.
function readPendingAuthDeepLink(isOnAuthRoute: boolean) {
  if (!isOnAuthRoute) return false;
  return getAuthDeepLink(readAuthNext()) !== null;
}

function readAuthNext() {
  return new URLSearchParams(window.location.search).get("next");
}

// Long enough for a full-page navigation (a login reloads the page) to finish.
const AUTH_ROUTE_FALLBACK_MS = 3_000;

export default function OnboardingProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [state, setState] = useState<UserOnboarding | null>(null);
  const [step, setStep] = useState(1);
  // Only a re-run trigger for the init effect, which reads the live URL.
  const [authDeepLinkPending, setAuthDeepLinkPending] = useState(false);
  const hasInitialized = useRef(false);
  const latestCheck = useRef(0);
  const isMounted = useRef(true);
  const pendingUpdatesRef = useRef<Set<Promise<void>>>(new Set());
  const { toast } = useToast();

  const api = useBackendAPI();
  const pathname = usePathname();
  const latestPathname = useRef(pathname);
  const [initRetry, setInitRetry] = useState(0);
  const router = useRouter();
  const { isLoggedIn, user } = useAuth();
  const userID = user?.id ?? null;
  const latestUserID = useRef(userID);
  const { isOnboardingComplete, setCompletion } = useOnboardingCompletion({
    isLoggedIn,
    userID,
  });

  useOnboardingTimezoneDetection();

  // Another account signed in on this tab needs its own answer. Updated at
  // commit, so an answer arriving straight after the switch can't miss it.
  useLayoutEffect(() => {
    if (latestUserID.current !== userID) hasInitialized.current = false;
    latestUserID.current = userID;
  }, [userID]);

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

  const {
    isOnOnboardingRoute,
    isOnAuthRoute,
    isExempt: isOnExemptRoute,
  } = getOnboardingRoute(pathname);

  const fetchOnboarding = useCallback(async () => {
    const requestUserID = latestUserID.current;
    const onboarding = await resolveResponse(getV1OnboardingState());
    const processedOnboarding = fromBackendUserOnboarding(onboarding);
    // An answer for an account that has since signed out is not this one's.
    if (isMounted.current && latestUserID.current === requestUserID) {
      // A websocket `onboarding` notification refetches even when nothing
      // changed; keeping the old object spares every consumer a re-render.
      setState((prev) =>
        isEqual(prev, processedOnboarding) ? prev : processedOnboarding,
      );
      // Only ever upgrades; "not complete" comes from the init's check.
      if (hasCompletedOnboarding(processedOnboarding.completedSteps)) {
        setCompletion((prev) =>
          withCompletion(prev, { userID: requestUserID, isCompleted: true }),
        );
      }
    }
    return processedOnboarding;
  }, [setCompletion]);

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

  // An unfinished user who left the wizard (Back, a link) is sent back on
  // their next page, as a full page load would.
  const isKnownUnfinished = isOnboardingComplete === false;
  useEffect(() => {
    if (isKnownUnfinished) hasInitialized.current = false;
  }, [isKnownUnfinished, pathname]);

  // At commit, with the URL: a navigation's passive effects can run a task
  // later, and an answer landing in between must see the new page.
  useLayoutEffect(() => {
    latestPathname.current = pathname;
  }, [pathname]);

  useEffect(() => {
    // Prevent multiple initializations. Skipped routes don't mark
    // hasInitialized, so init re-runs on the next pathname change.
    if (hasInitialized.current || !isLoggedIn || isOnExemptRoute) {
      return;
    }

    // Defer to the auth-page's own deep-link redirect (`?next=…`) instead of
    // racing it. Don't mark hasInitialized so the next pathname change
    // (after the auth page redirects) gets us properly initialized.
    const hasPendingAuthDeepLink = readPendingAuthDeepLink(isOnAuthRoute);
    if (hasPendingAuthDeepLink) {
      return;
    }

    hasInitialized.current = true;
    const checkID = ++latestCheck.current;

    async function initializeOnboarding() {
      const requestUserID = latestUserID.current;
      // Re-checks can overlap and answer out of order. Only the newest decides,
      // and only for the account it asked about.
      function isOutdated() {
        return (
          latestCheck.current !== checkID ||
          latestUserID.current !== requestUserID
        );
      }
      try {
        const { is_completed } = await retryTransient(() =>
          resolveResponse(getV1CheckIfOnboardingIsCompleted()),
        );
        if (isOutdated()) return;

        // A "not finished" read in the wizard is out of date if the user has
        // since finished it, so once they have left it their state decides.
        let onboarding: UserOnboarding | null = null;
        const hasLeftWizard =
          isOnOnboardingRoute &&
          !getOnboardingRoute(latestPathname.current).isOnOnboardingRoute;
        if (!is_completed && hasLeftWizard) {
          // Unreadable, it goes to the catch, which asks again rather than
          // guess: a finished user must not be sent back into the wizard.
          onboarding = await retryTransient(fetchOnboarding);
          if (isOutdated()) return;
        }
        const isCompleted =
          is_completed ||
          (onboarding !== null &&
            hasCompletedOnboarding(onboarding.completedSteps));
        if (isMounted.current) {
          setCompletion((prev) =>
            withCompletion(prev, { userID: requestUserID, isCompleted }),
          );
        }

        // The user may have moved on while this was in flight (the init
        // skips their new page while it runs), so the page they are on now
        // decides, with the answer already in hand.
        const current = getOnboardingRoute(latestPathname.current);
        const redirectTarget = current.isExempt
          ? null
          : decideOnboardingRedirect({
              isCompleted,
              isOnOnboardingRoute: current.isOnOnboardingRoute,
              isOnAuthRoute: current.isOnAuthRoute,
              hasPendingAuthDeepLink: readPendingAuthDeepLink(
                current.isOnAuthRoute,
              ),
            });
        // An admin impersonating an unfinished user is never sent into the
        // target's wizard; the answer is the target's, not theirs. A sign-in
        // page only shows its loader, so they still have to leave it.
        const isAdminViewingTarget =
          redirectTarget === "/onboarding" && hasImpersonation();
        const destination = !isAdminViewingTarget
          ? redirectTarget
          : current.isOnAuthRoute
            ? "/home"
            : null;
        if (destination) {
          // The next page reads the onboarding state this return skips.
          hasInitialized.current = false;
          // Leaving an auth page keeps a campaign link's tracking params.
          router.replace(
            current.isOnAuthRoute
              ? withAuthTracking(destination, readAuthNext())
              : destination,
          );
          return;
        }

        onboarding ??= await retryTransient(fetchOnboarding);
        if (isOutdated()) return;

        const pathNow = latestPathname.current;
        if (
          getOnboardingRoute(pathNow).isOnOnboardingRoute &&
          shouldRedirectFromOnboarding(onboarding.completedSteps, pathNow)
        ) {
          router.replace("/home");
        }
      } catch (error) {
        if (isOutdated()) return;

        // Only this check moves a signed-in user off a sign-in page's loader.
        // Home's own check takes over from there, and the prompts on Home
        // stay hidden until it answers.
        function leaveAuthRouteAfterWait() {
          setTimeout(() => {
            const { isOnAuthRoute } = getOnboardingRoute(
              latestPathname.current,
            );
            if (
              isMounted.current &&
              !isOutdated() &&
              isOnAuthRoute &&
              !readPendingAuthDeepLink(isOnAuthRoute)
            ) {
              router.replace(withAuthTracking("/home", readAuthNext()));
            }
          }, AUTH_ROUTE_FALLBACK_MS);
        }
        const isOnAuthRouteNow = getOnboardingRoute(
          latestPathname.current,
        ).isOnAuthRoute;

        if (error instanceof ApiError && error.status === 401) {
          hasInitialized.current = false;
          // A dead session is the auth store's to end, which brings back the
          // sign-in form. Home is only for a session it still accepts.
          if (isOnAuthRouteNow) {
            const isValid = await useAuthStore
              .getState()
              .validateSession({ force: true });
            if (isValid) leaveAuthRouteAfterWait();
          }
          return;
        }

        if (isOnAuthRouteNow) leaveAuthRouteAfterWait();

        // A fetch that still can't get through after the retries means the
        // user is offline or navigating away, not looking at a broken
        // onboarding. Try again on the next route change instead of alarming
        // them with a destructive toast.
        if (isAbortError(error) || isNetworkError(error)) {
          console.warn("Onboarding initialization deferred:", error);
          hasInitialized.current = false;
          // A route change that landed while this was in flight was skipped
          // by the guard above, so run again for it now.
          if (latestPathname.current !== pathname) setInitRetry((n) => n + 1);
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
    authDeepLinkPending,
    router,
    isLoggedIn,
    pathname,
    isOnExemptRoute,
    initRetry,
    userID,
    setCompletion,
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

  const value = useMemo(
    () => ({
      state,
      isOnboardingComplete,
      updateState,
      step,
      setStep,
      completeStep,
    }),
    [state, isOnboardingComplete, updateState, step, completeStep],
  );

  return (
    <OnboardingContext.Provider value={value}>
      {isOnAuthRoute && (
        <AuthDeepLinkWatcher onChange={setAuthDeepLinkPending} />
      )}
      {children}
    </OnboardingContext.Provider>
  );
}
