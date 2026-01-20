"use client";

import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { getHomepageRoute } from "@/lib/constants";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  Flag,
  type FlagValues,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";
import { useFlags } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  getGreetingName,
  getQuickActions,
} from "./helpers";

export function useCopilotHome() {
  const router = useRouter();
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const [value, setValue] = useState("");
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const flags = useFlags<FlagValues>();
  const homepageRoute = getHomepageRoute(isChatEnabled);
  const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const isLaunchDarklyConfigured = envEnabled && Boolean(clientId);
  const isFlagReady =
    !isLaunchDarklyConfigured || flags[Flag.CHAT] !== undefined;
  
    const greetingName = useMemo(
    function getName() {
      return getGreetingName(user);
    },
    [user],
  );

  const quickActions = useMemo(function getActions() {
    return getQuickActions();
  }, []);

  useEffect(
    function ensureAccess() {
      if (!isFlagReady) return;
      if (isChatEnabled === false) {
        router.replace(homepageRoute);
      }
    },
    [homepageRoute, isChatEnabled, isFlagReady, router],
  );

  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setValue(event.target.value);
  }

  async function createSessionAndNavigate(prompt: string) {
    try {
      const response = await postV2CreateSession({ body: JSON.stringify({}) });
      if (response.status === 200 && response.data) {
        try {
          sessionStorage.setItem("copilot_initial_prompt", prompt);
        } catch {
          // Ignore storage errors (private mode, etc.)
        }
        router.push(`/copilot/chat?sessionId=${response.data.id}`);
      }
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!value.trim()) return;
    await createSessionAndNavigate(value.trim());
  }

  async function handleKeyDown(
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (event.key !== "Enter") return;
    if (event.shiftKey) return;
    event.preventDefault();
    if (!value.trim()) return;
    await createSessionAndNavigate(value.trim());
  }

  async function handleQuickAction(action: string) {
    await createSessionAndNavigate(action);
  }

  return {
    greetingName,
    value,
    quickActions,
    isFlagReady,
    isChatEnabled,
    isLoggedIn,
    isUserLoading,
    handleChange,
    handleSubmit,
    handleKeyDown,
    handleQuickAction,
  };
}
