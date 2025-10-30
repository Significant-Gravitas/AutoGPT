import { useEffect, useState, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import { useChatSession } from "@/hooks/useChatSession";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

interface UseChatPageResult {
  session: ReturnType<typeof useChatSession>["session"];
  messages: ReturnType<typeof useChatSession>["messages"];
  isLoading: boolean;
  isCreating: boolean;
  error: Error | null;
  createSession: () => Promise<string>;
  refreshSession: () => Promise<void>;
  clearSession: () => void;
  sessionId: string | null;
}

export function useChatPage(): UseChatPageResult {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlSessionId = searchParams.get("session");
  const { user, isUserLoading } = useSupabase();
  const [_isOnline, setIsOnline] = useState(true);
  const claimingSessionRef = useRef<string | null>(null);
  const claimedSessionsRef = useRef<Set<string>>(new Set());
  const recoveringFromErrorRef = useRef<boolean>(false);

  const {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    refreshSession,
    clearSession: clearSessionBase,
    claimSession,
  } = useChatSession({
    urlSessionId,
    autoCreate: false, // We'll manually create when needed
  });

  // Auto-create session if none exists or if the existing one failed to load
  useEffect(
    function autoCreateSession() {
      // Skip if we're recovering from an error
      if (recoveringFromErrorRef.current) {
        return;
      }

      // Simple case: no URL session, no current session, not loading
      if (!urlSessionId && !session && !isLoading && !isCreating && !error) {
        createSession().catch((err) => {
          console.error("Failed to auto-create session:", err);
        });
        return;
      }

      // Error case: session failed to load (404), create a new one
      // Only do this once - don't create multiple times
      if (error && !isCreating && !session && !claimingSessionRef.current) {
        claimingSessionRef.current = "creating"; // Use ref as a lock
        recoveringFromErrorRef.current = true; // Set recovery flag

        createSession()
          .then((_newSessionId) => {
            claimingSessionRef.current = null;
            // Remove old session from URL if present, new session is in localStorage
            if (urlSessionId) {
              router.replace("/chat");
            }

            // Clear recovery flag after a short delay to allow state to settle
            setTimeout(() => {
              recoveringFromErrorRef.current = false;
            }, 500);
          })
          .catch((err) => {
            console.error("Failed to create new session after error:", err);
            claimingSessionRef.current = null;
            recoveringFromErrorRef.current = false;
          });
      }
    },
    [
      urlSessionId,
      session,
      isLoading,
      isCreating,
      error,
      createSession,
      router,
    ],
  );

  // Auto-claim anonymous sessions when user logs in
  useEffect(
    function autoClaimSession() {
      // Skip if no user or still loading
      if (!user || isUserLoading || !session?.id) {
        return;
      }

      // Skip if we're recovering from an error
      if (recoveringFromErrorRef.current) {
        return;
      }

      // Anonymous session that needs claiming
      if (!session.user_id) {
        // Prevent duplicate claims
        if (
          claimingSessionRef.current === session.id ||
          claimedSessionsRef.current.has(session.id)
        ) {
          return;
        }

        claimingSessionRef.current = session.id;

        claimSession(session.id)
          .then(() => {
            claimedSessionsRef.current.add(session.id);
            claimingSessionRef.current = null;
          })
          .catch((err) => {
            console.error("Failed to auto-claim session:", err);
            claimingSessionRef.current = null;

            // If session doesn't exist (404) or belongs to another user, create a new one
            if (err?.status === 404 || err?.response?.status === 404) {
              // Set recovery flag to prevent effect from running again during recovery
              recoveringFromErrorRef.current = true;

              clearSessionBase();
              createSession()
                .then((_newSessionId) => {
                  // Remove old session from URL, new session is in localStorage
                  router.replace("/chat");

                  // Clear recovery flag after a short delay to allow state to settle
                  setTimeout(() => {
                    recoveringFromErrorRef.current = false;
                  }, 500);
                })
                .catch((createErr) => {
                  console.error("Failed to create new session:", createErr);
                  recoveringFromErrorRef.current = false;
                });
            }
          });
      }
      // Session belongs to a different user
      else if (session.user_id !== user.id) {
        // Set recovery flag to prevent effect from running again during recovery
        recoveringFromErrorRef.current = true;

        clearSessionBase();
        createSession()
          .then((_newSessionId) => {
            // Remove old session from URL, new session is in localStorage
            router.replace("/chat");

            // Clear recovery flag after a short delay to allow state to settle
            setTimeout(() => {
              recoveringFromErrorRef.current = false;
            }, 500);
          })
          .catch((createErr) => {
            console.error("Failed to create new session:", createErr);
            recoveringFromErrorRef.current = false;
          });
      }
    },
    [
      user,
      isUserLoading,
      session?.id,
      session?.user_id,
      claimSession,
      clearSessionBase,
      createSession,
      router,
    ],
  );

  // Monitor online/offline status
  useEffect(function monitorNetworkStatus() {
    function handleOnline() {
      setIsOnline(true);
      toast.success("Connection restored", {
        description: "You're back online",
      });
    }

    function handleOffline() {
      setIsOnline(false);
      toast.error("You're offline", {
        description: "Check your internet connection",
      });
    }

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    // Check initial status
    setIsOnline(navigator.onLine);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const clearSession = () => {
    clearSessionBase();
    // Remove session from URL
    router.push("/chat");
  };

  return {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    refreshSession,
    clearSession,
    sessionId: session?.id || null,
  };
}
