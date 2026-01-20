"use client";

import { getHomepageRoute } from "@/lib/constants";
import {
  Flag,
  type FlagValues,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";
import { useFlags } from "launchdarkly-react-client-sdk";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

export function useCopilotChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const flags = useFlags<FlagValues>();
  const homepageRoute = getHomepageRoute(isChatEnabled);
  const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const isLaunchDarklyConfigured = envEnabled && Boolean(clientId);
  const isFlagReady =
    !isLaunchDarklyConfigured || flags[Flag.CHAT] !== undefined;

  const sessionId = searchParams.get("sessionId");
  const prompt = searchParams.get("prompt");
  const [storedPrompt, setStoredPrompt] = useState<string | null>(null);

  useEffect(
    function loadStoredPrompt() {
      if (prompt) return;
      try {
        const storedValue = sessionStorage.getItem("copilot_initial_prompt");
        if (!storedValue) return;
        sessionStorage.removeItem("copilot_initial_prompt");
        setStoredPrompt(storedValue);
      } catch {
        // Ignore storage errors (private mode, etc.)
      }
    },
    [prompt],
  );

  useEffect(
    function guardAccess() {
      if (!isFlagReady) return;
      if (isChatEnabled === false) {
        router.replace(homepageRoute);
      }
    },
    [homepageRoute, isChatEnabled, isFlagReady, router],
  );

  return {
    isFlagReady,
    isChatEnabled,
    sessionId,
    prompt: prompt ?? storedPrompt,
  };
}
