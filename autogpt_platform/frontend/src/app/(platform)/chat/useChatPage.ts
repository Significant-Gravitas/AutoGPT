import { useEffect, useState, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import { useChatSession } from "@/hooks/useChatSession";

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
  const [isOnline, setIsOnline] = useState(true);
  const hasCreatedSessionRef = useRef(false);

  const {
    session,
    sessionId: sessionIdFromHook,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    refreshSession,
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

  // Note: Session claiming is handled explicitly by UI components when needed
  // - Locally created sessions: backend sets user_id from JWT automatically
  // - URL sessions: claiming happens in specific user flows, not automatically

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
    // Remove session from URL
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
