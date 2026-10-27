"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useAuth } from "@/lib/auth/hooks/useAuth";
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
 * 4. The user has notifications switched on
 *
 * Switching notifications off, or logging out, unsubscribes both locally and
 * on the backend. The backend pushes to every subscription a user has, so
 * keeping one would keep the OS notifications coming.
 */
export function usePushNotifications() {
  const { user } = useAuth();
  const isEnabled = useCopilotUIStore((s) => s.isNotificationsEnabled);
  const registeredRef = useRef(false);
  const wasAuthedRef = useRef(false);
  const pendingTeardownRef = useRef<Promise<void>>(Promise.resolve());
  const [renewCount, setRenewCount] = useState(0);

  useEffect(() => {
    if (typeof window === "undefined") return;

    let cancelled = false;

    // Each teardown queues behind the one before, so a setup waiting on the
    // latest has waited for all of them.
    function startTeardown() {
      pendingTeardownRef.current = pendingTeardownRef.current
        .then(teardownPushSubscription)
        .catch((error) =>
          console.error("Push notification teardown failed:", error),
        );
    }

    if (user && isEnabled) {
      wasAuthedRef.current = true;
      if (registeredRef.current) return;
      // Wait for an earlier teardown, or it could remove the subscription
      // this setup is about to make.
      pendingTeardownRef.current
        .then(() => setupPushSubscription(() => cancelled))
        .then((sent) => {
          if (sent && !cancelled) registeredRef.current = true;
        })
        .catch((error) =>
          console.error("Push notification setup failed:", error),
        );
      return () => {
        cancelled = true;
      };
    }

    if (user) {
      // Also runs on load, so a browser subscribed before this check existed
      // drops its subscription the next time the app opens.
      wasAuthedRef.current = true;
      registeredRef.current = false;
      startTeardown();
      return;
    }

    if (!wasAuthedRef.current) return;
    wasAuthedRef.current = false;
    registeredRef.current = false;
    startTeardown();
  }, [user, isEnabled, renewCount]);

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
