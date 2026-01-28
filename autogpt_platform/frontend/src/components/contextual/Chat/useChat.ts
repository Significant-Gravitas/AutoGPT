"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect, useRef, useState } from "react";
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
    startPollingForOperation,
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

  useEffect(
    function showLoaderWithDelay() {
      if (isLoading || isCreating) {
        const timer = setTimeout(() => setShowLoader(true), 300);
        return () => clearTimeout(timer);
      }
      setShowLoader(false);
    },
    [isLoading, isCreating],
  );

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
    startPollingForOperation,
  };
}
