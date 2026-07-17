"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useEffect } from "react";

/**
 * Forwards the user's notification toggle to the push service worker so that
 * disabling notifications also suppresses OS-level pushes (which arrive
 * directly via FCM and bypass any in-page gating).
 *
 * SW state resets on every restart, so the hook also re-sends on mount and
 * on `controllerchange` to re-sync after updates/installs.
 */
export function useReportNotificationsEnabled() {
  const isEnabled = useCopilotUIStore((s) => s.isNotificationsEnabled);

  useEffect(() => {
    if (typeof navigator === "undefined") return;
    if (!navigator.serviceWorker) return;

    function send() {
      navigator.serviceWorker.controller?.postMessage({
        type: "NOTIFICATIONS_ENABLED",
        value: isEnabled,
      });
    }

    send();
    navigator.serviceWorker.addEventListener("controllerchange", send);
    return () =>
      navigator.serviceWorker.removeEventListener("controllerchange", send);
  }, [isEnabled]);
}
