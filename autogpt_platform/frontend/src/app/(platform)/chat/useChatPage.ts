"use client";

import { useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { useChatSession } from "@/app/(platform)/chat/useChatSession";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useChatStream } from "@/app/(platform)/chat/useChatStream";

export function useChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlSessionId =
    searchParams.get("session_id") || searchParams.get("session");
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
    autoCreate: false,
  });

  useEffect(
    function autoCreateSession() {
      if (
        !urlSessionId &&
        !hasCreatedSessionRef.current &&
        !isCreating &&
        !sessionIdFromHook
      ) {
        hasCreatedSessionRef.current = true;
        createSession().catch((_err) => {
          hasCreatedSessionRef.current = false;
        });
      }
    },
    [urlSessionId, isCreating, sessionIdFromHook, createSession],
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
    sessionId: sessionIdFromHook,
  };
}
