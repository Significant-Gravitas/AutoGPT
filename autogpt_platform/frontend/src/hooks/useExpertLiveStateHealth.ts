"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import * as Sentry from "@sentry/nextjs";
import { useEffect, useState } from "react";

export type ExpertSurface = "copilot" | "home" | "team";
export type LiveStateHealth = "connecting" | "live" | "polling";

const surfaceFlags = new Map<ExpertSurface, boolean>();
const lastTransportReport = new Map<ExpertSurface, LiveStateHealth>();

interface Args {
  surface: ExpertSurface;
  hireExpertsEnabled: boolean;
  onFallbackRefresh?: () => void | Promise<unknown>;
}

export function useExpertLiveStateHealth({
  surface,
  hireExpertsEnabled,
  onFallbackRefresh,
}: Args) {
  const api = useBackendAPI();
  const [health, setHealth] = useState<LiveStateHealth>("connecting");

  useEffect(() => {
    surfaceFlags.set(surface, hireExpertsEnabled);
    reportFlagParity();
    let disposed = false;
    let fallbackTimer: ReturnType<typeof setTimeout> | null = null;
    let detachConnect = () => {};
    let detachDisconnect = () => {};
    let detachNotification = () => {};

    function activatePolling() {
      if (disposed) return;
      setHealth("polling");
      reportTransport(surface, "polling");
    }

    function activateLive() {
      if (disposed) return;
      if (fallbackTimer !== null) clearTimeout(fallbackTimer);
      fallbackTimer = null;
      setHealth("live");
      reportTransport(surface, "live");
      void onFallbackRefresh?.();
    }

    if (hireExpertsEnabled) {
      fallbackTimer = setTimeout(activatePolling, 4_000);
      detachConnect = api.onWebSocketConnect(activateLive);
      detachDisconnect = api.onWebSocketDisconnect(activatePolling);
      detachNotification = api.onWebSocketMessage("notification", () => {
        void onFallbackRefresh?.();
      });
      void api.connectWebSocket().catch(activatePolling);
    } else {
      setHealth("connecting");
    }

    return () => {
      disposed = true;
      detachConnect();
      detachDisconnect();
      detachNotification();
      surfaceFlags.delete(surface);
      lastTransportReport.delete(surface);
      if (fallbackTimer !== null) clearTimeout(fallbackTimer);
    };
  }, [api, hireExpertsEnabled, onFallbackRefresh, surface]);

  useEffect(() => {
    if (health !== "polling") return;
    void onFallbackRefresh?.();
    const timer = setInterval(() => {
      void onFallbackRefresh?.();
    }, 5_000);
    return () => clearInterval(timer);
  }, [health, onFallbackRefresh]);

  return health;
}

function reportTransport(surface: ExpertSurface, health: LiveStateHealth) {
  if (lastTransportReport.get(surface) === health) return;
  lastTransportReport.set(surface, health);
  if (health !== "polling") return;
  Sentry.captureMessage("expert_live_state_polling_fallback", {
    level: "warning",
    tags: {
      surface,
      live_state_transport: health,
      hire_experts: String(surfaceFlags.get(surface) ?? false),
    },
  });
}

function reportFlagParity() {
  const entries = [...surfaceFlags.entries()];
  if (entries.length < 2) return;
  const values = new Set(entries.map(([, value]) => value));
  if (values.size < 2) return;
  Sentry.captureMessage("hire_experts_surface_flag_mismatch", {
    level: "error",
    contexts: { surfaces: Object.fromEntries(entries) },
  });
}

export const liveStateHealthTestUtils = {
  reset() {
    surfaceFlags.clear();
    lastTransportReport.clear();
  },
};
