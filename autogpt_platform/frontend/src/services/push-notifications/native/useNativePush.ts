"use client";

import { useEffect } from "react";
import { create } from "zustand";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { hasNativePush, requestNativePush } from "./bridge";

interface PushState {
  available: boolean;
  enabled: boolean;
  busy: boolean;
  error: string | null;
}

export const useNativePushState = create<PushState>(() => ({
  available: false,
  enabled: false,
  busy: false,
  error: null,
}));
let generation = 0;
let operations: Promise<void> = Promise.resolve();

export function updateNativePush(
  accountID: string,
  action: "status" | "enable" | "disable",
) {
  const current = generation;
  const operation = operations.then(async () => {
    if (current !== generation) return;
    useNativePushState.setState({ busy: true, error: null });
    try {
      if (action === "disable") await post("remove", {});
      const result = await requestNativePush(action, accountID);
      if (current !== generation) return;
      if (
        result.permission === "granted" &&
        result.token &&
        result.provider &&
        result.environment &&
        result.binding_id
      ) {
        const configResponse = await fetch(
          "/api/proxy/api/push/native/config",
          { signal: AbortSignal.timeout(10_000) },
        );
        const config: unknown = configResponse.ok
          ? await configResponse.json()
          : null;
        if (
          !config ||
          typeof config !== "object" ||
          !(result.provider in config) ||
          Reflect.get(config, result.provider) !== true
        ) {
          throw new Error(
            "Push notifications aren't configured on this server yet.",
          );
        }
        if (current !== generation) return;
        await post("", {
          provider: result.provider,
          token: result.token,
          environment: result.environment,
          binding_id: result.binding_id,
          expected_user_id: accountID,
        });
        if (current === generation)
          useNativePushState.setState({ enabled: true });
      } else {
        useNativePushState.setState({ enabled: false });
        if (action === "enable")
          throw new Error(
            result.permission === "denied"
              ? "Allow AutoGPT notifications in your device settings, then try again."
              : "Push notifications aren't configured for this app build yet.",
          );
      }
    } catch (error) {
      if (current === generation)
        useNativePushState.setState({
          error:
            error instanceof Error
              ? error.message
              : "Couldn't update notifications. Try again.",
        });
    } finally {
      if (current === generation) useNativePushState.setState({ busy: false });
    }
  });
  operations = operation.catch(() => {});
  return operation;
}

async function post(path: string, body: object) {
  const response = await fetch(
    `/api/auth/mobile/push${path ? `/${path}` : ""}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    },
  );
  if (!response.ok)
    throw new Error(
      "Couldn't save notification settings. Check your connection and sign-in, then try again.",
    );
}

export function useNativePush() {
  const { user, isUserLoading } = useAuth();
  const accountID = user?.id;
  useEffect(() => {
    if (isUserLoading) return;
    generation++;
    const available = hasNativePush();
    useNativePushState.setState({
      available,
      enabled: false,
      busy: false,
      error: null,
    });
    if (!available) return;
    if (!accountID) {
      void requestNativePush("disable", "").catch(() => {});
      return;
    }
    function refresh() {
      if (accountID && !useNativePushState.getState().busy)
        void updateNativePush(accountID, "status");
    }
    function visibility() {
      if (document.visibilityState === "visible") refresh();
    }
    refresh();
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      generation++;
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [accountID, isUserLoading]);
}
