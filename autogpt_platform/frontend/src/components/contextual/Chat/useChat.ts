"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatSession } from "./useChatSession";
import { useChatStream } from "./useChatStream";

interface UseChatArgs {
  urlSessionId?: string | null;
}

export function useChat({ urlSessionId }: UseChatArgs = {}) {
  const hasClaimedSessionRef = useRef(false);
  const { user } = useSupabase();
  const { sendMessage: sendStreamMessage } = useChatStream();
  const [showLoader, setShowLoader] = useState(false);
  const {
    session,
    sessionId: sessionIdFromHook,
    messages,
    isLoading,
    isCreating,
    error,
    isSessionNotFound,
    createSession,
    claimSession,
    clearSession: clearSessionBase,
    loadSession,
  } = useChatSession({
    urlSessionId,
    autoCreate: false,
  });

  useEffect(
    function autoClaimSession() {
      if (
        session &&
        !session.user_id &&
        user &&
        !hasClaimedSessionRef.current &&
        !isLoading &&
        sessionIdFromHook
      ) {
        hasClaimedSessionRef.current = true;
        claimSession(sessionIdFromHook)
          .then(() => {
            sendStreamMessage(
              sessionIdFromHook,
              "User has successfully logged in.",
              () => {},
              false,
            ).catch(() => {});
          })
          .catch(() => {
            hasClaimedSessionRef.current = false;
          });
      }
    },
    [
      session,
      user,
      isLoading,
      sessionIdFromHook,
      claimSession,
      sendStreamMessage,
    ],
  );

  useEffect(() => {
    if (isLoading || isCreating) {
      const timer = setTimeout(() => {
        setShowLoader(true);
      }, 300);
      return () => clearTimeout(timer);
    } else {
      setShowLoader(false);
    }
  }, [isLoading, isCreating]);

  useEffect(function monitorNetworkStatus() {
    function handleOnline() {
      toast.success("Connection restored", {
        description: "You're back online",
      });
    }

    function handleOffline() {
      toast.error("You're offline", {
        description: "Check your internet connection",
      });
    }

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  function clearSession() {
    clearSessionBase();
    hasClaimedSessionRef.current = false;
  }

  return {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    isSessionNotFound,
    createSession,
    clearSession,
    loadSession,
    sessionId: sessionIdFromHook,
    showLoader,
  };
}
