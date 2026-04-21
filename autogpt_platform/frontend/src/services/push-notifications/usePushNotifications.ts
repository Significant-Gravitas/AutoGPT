"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect, useRef, useState } from "react";
import { fetchVapidPublicKey, sendSubscriptionToServer } from "./api";
import {
  isPushSupported,
  registerServiceWorker,
  subscribeToPush,
} from "./registration";
import type { PushSWMessage } from "./types";

/**
 * Registers the push service worker and sends the subscription to the backend.
 * Should be mounted once at the platform layout level.
 *
 * Only activates when:
 * 1. The browser supports Push API + Service Workers
 * 2. Notification permission is already "granted" (requested by copilot flow)
 * 3. The user is authenticated
 */
export function usePushNotifications() {
  const { user } = useSupabase();
  const registeredRef = useRef(false);
  const [renewCount, setRenewCount] = useState(0);

  useEffect(() => {
    if (!user || registeredRef.current) return;
    if (typeof window === "undefined") return;

    async function setup() {
      if (!isPushSupported()) return;
      if (Notification.permission !== "granted") return;

      const registration = await registerServiceWorker();
      if (!registration) return;

      await navigator.serviceWorker.ready;

      const vapidKey =
        process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY ||
        (await fetchVapidPublicKey());
      if (!vapidKey) return;

      const subscription = await subscribeToPush(registration, vapidKey);
      if (!subscription) return;

      const sent = await sendSubscriptionToServer(subscription);
      if (sent) {
        registeredRef.current = true;
      }
    }

    setup().catch((error) =>
      console.error("Push notification setup failed:", error),
    );
  }, [user, renewCount]);

  // Listen for subscription change events from the service worker
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
