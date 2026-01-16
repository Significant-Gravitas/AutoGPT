"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useChatSession } from "./useChatSession";
import { useChatStream } from "./useChatStream";

export function useChat() {
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
    claimSession,
    clearSession: clearSessionBase,
    loadSession,
  } = useChatSession({
    urlSessionId: null,
    autoCreate: false,
  });

  useEffect(
    function autoCreateSession() {
      if (!hasCreatedSessionRef.current && !isCreating && !sessionIdFromHook) {
        hasCreatedSessionRef.current = true;
        createSession().catch((_err) => {
          hasCreatedSessionRef.current = false;
        });
      }
    },
    [isCreating, sessionIdFromHook, createSession],
  );

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
    hasCreatedSessionRef.current = false;
    hasClaimedSessionRef.current = false;
  }

  return {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    clearSession,
    loadSession,
    sessionId: sessionIdFromHook,
  };
}
