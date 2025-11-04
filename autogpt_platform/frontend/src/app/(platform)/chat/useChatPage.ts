import { useEffect, useState, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import { useChatSession } from "@/hooks/useChatSession";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useChatStream } from "@/hooks/useChatStream";

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
  // Support both 'session' and 'session_id' query parameters
  const urlSessionId =
    searchParams.get("session_id") || searchParams.get("session");
  const [isOnline, setIsOnline] = useState(true);
  const hasCreatedSessionRef = useRef(false);
  const hasClaimedSessionRef = useRef(false);
  const { user } = useSupabase();
  const { sendMessage: sendStreamMessage } = useChatStream();

  const {
    session,
    sessionId: sessionIdFromHook,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    refreshSession,
    claimSession,
    clearSession: clearSessionBase,
  } = useChatSession({
    urlSessionId,
    autoCreate: false, // We'll manually create when needed
  });

  // Auto-create session ONLY if there's no URL session
  // If URL session exists, GET query will fetch it automatically
  useEffect(
    function autoCreateSession() {
      // Only create if:
      // 1. No URL session (not loading someone else's session)
      // 2. Haven't already created one this mount
      // 3. Not currently creating
      // 4. We don't already have a sessionId
      if (
        !urlSessionId &&
        !hasCreatedSessionRef.current &&
        !isCreating &&
        !sessionIdFromHook
      ) {
        console.log("[autoCreateSession] Creating new session");
        hasCreatedSessionRef.current = true;
        createSession().catch((err) => {
          console.error("[autoCreateSession] Failed to create session:", err);
          hasCreatedSessionRef.current = false; // Reset on error to allow retry
        });
      } else if (sessionIdFromHook) {
        console.log(
          "[autoCreateSession] Skipping - already have sessionId:",
          sessionIdFromHook,
        );
      }
    },
    [urlSessionId, isCreating, sessionIdFromHook, createSession],
  );

  // Auto-claim session if user is logged in and session has no user_id
  useEffect(
    function autoClaimSession() {
      // Only claim if:
      // 1. We have a session loaded
      // 2. Session has no user_id (anonymous session)
      // 3. User is logged in
      // 4. Haven't already claimed this session
      // 5. Not currently loading
      if (
        session &&
        !session.user_id &&
        user &&
        !hasClaimedSessionRef.current &&
        !isLoading &&
        sessionIdFromHook
      ) {
        console.log("[autoClaimSession] Claiming anonymous session for user");
        hasClaimedSessionRef.current = true;
        claimSession(sessionIdFromHook)
          .then(() => {
            console.log(
              "[autoClaimSession] Session claimed successfully, sending login notification",
            );
            // Send login notification message to backend after successful claim
            // This notifies the agent that the user has logged in
            sendStreamMessage(
              sessionIdFromHook,
              "User has successfully logged in.",
              () => {
                // Empty chunk handler - we don't need to process responses for this system message
              },
              false, // isUserMessage = false
            ).catch((err) => {
              console.error(
                "[autoClaimSession] Failed to send login notification:",
                err,
              );
            });
          })
          .catch((err) => {
            console.error("[autoClaimSession] Failed to claim session:", err);
            hasClaimedSessionRef.current = false; // Reset on error to allow retry
          });
      }
    },
    [session, user, isLoading, sessionIdFromHook, claimSession, sendStreamMessage],
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

  function clearSession() {
    clearSessionBase();
    // Reset the created session flag so a new session can be created
    hasCreatedSessionRef.current = false;
    hasClaimedSessionRef.current = false;
    // Remove session from URL and trigger new session creation
    router.push("/chat");
  }

  return {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    refreshSession,
    clearSession,
    sessionId: sessionIdFromHook, // Use direct sessionId from hook, not derived from session.id
  };
}
