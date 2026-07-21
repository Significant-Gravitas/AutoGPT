"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect, useRef, useState } from "react";
import { setupPushSubscription, teardownPushSubscription } from "./helpers";
import type { PushSWMessage } from "./types";

/**
 * Registers the push service worker and sends the subscription to the backend.
 * Should be mounted once at the platform layout level.
 *
 * Only activates when:
 * 1. The browser supports Push API + Service Workers
 * 2. Notification permission is already "granted" (requested by copilot flow)
 * 3. The user is authenticated
 *
 * On logout, unsubscribes both locally and on the backend so the previous
 * user doesn't keep receiving OS notifications.
 */
export function usePushNotifications() {
  const { user } = useSupabase();
  const registeredRef = useRef(false);
  const wasAuthedRef = useRef(false);
  const [renewCount, setRenewCount] = useState(0);

  useEffect(() => {
    if (typeof window === "undefined") return;

    if (user) {
      wasAuthedRef.current = true;
      if (registeredRef.current) return;
      setupPushSubscription()
        .then((sent) => {
          if (sent) registeredRef.current = true;
        })
        .catch((error) =>
          console.error("Push notification setup failed:", error),
        );
      return;
    }

    if (!wasAuthedRef.current) return;
    wasAuthedRef.current = false;
    registeredRef.current = false;
    teardownPushSubscription().catch((error) =>
      console.error("Push notification teardown failed:", error),
    );
  }, [user, renewCount]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    function handleMessage(event: MessageEvent<PushSWMessage>) {
      if (event.data?.type === "PUSH_SUBSCRIPTION_CHANGED") {
        registeredRef.current = false;
        setRenewCount((c) => c + 1);
      }
    }

    navigator.serviceWorker.addEventListener("message", handleMessage);
    return () =>
      navigator.serviceWorker.removeEventListener("message", handleMessage);
  }, []);
}
