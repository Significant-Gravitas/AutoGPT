import {
  getGetV2ListSessionsQueryKey,
  postV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getHomepageRoute } from "@/lib/constants";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  Flag,
  type FlagValues,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useFlags } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useCopilotStore } from "./copilot-page-store";
import { getGreetingName, getQuickActions } from "./helpers";
import { useCopilotSessionId } from "./useCopilotSessionId";

export function useCopilotPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const { toast } = useToast();

  const { urlSessionId, setUrlSessionId } = useCopilotSessionId();
  const setIsStreaming = useCopilotStore((s) => s.setIsStreaming);

  const isChatEnabled = useGetFlag(Flag.CHAT);
  const flags = useFlags<FlagValues>();
  const homepageRoute = getHomepageRoute(isChatEnabled);
  const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const isLaunchDarklyConfigured = envEnabled && Boolean(clientId);
  const isFlagReady =
    !isLaunchDarklyConfigured || flags[Flag.CHAT] !== undefined;

  const [isCreating, setIsCreating] = useState(false);
  const initialPromptsRef = useRef<Record<string, string>>({});

  const greetingName = getGreetingName(user);
  const quickActions = getQuickActions();

  const hasSession = Boolean(urlSessionId);
  const initialPrompt = urlSessionId
    ? initialPromptsRef.current[urlSessionId]
    : undefined;

  useEffect(() => {
    if (!isFlagReady) return;
    if (isChatEnabled === false) {
      router.replace(homepageRoute);
    }
  }, [homepageRoute, isChatEnabled, isFlagReady, router]);

  async function startChatWithPrompt(prompt: string) {
    if (!prompt?.trim()) return;
    if (isCreating) return;

    const trimmedPrompt = prompt.trim();
    setIsCreating(true);

    try {
      const sessionResponse = await postV2CreateSession({
        body: JSON.stringify({}),
      });

      if (sessionResponse.status !== 200 || !sessionResponse.data?.id) {
        throw new Error("Failed to create session");
      }

      const sessionId = sessionResponse.data.id;
      initialPromptsRef.current[sessionId] = trimmedPrompt;

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });

      await setUrlSessionId(sessionId, { shallow: false });
    } catch (error) {
      console.error("[CopilotPage] Failed to start chat:", error);
      toast({ title: "Failed to start chat", variant: "destructive" });
      Sentry.captureException(error);
    } finally {
      setIsCreating(false);
    }
  }

  function handleQuickAction(action: string) {
    startChatWithPrompt(action);
  }

  function handleSessionNotFound() {
    router.replace("/copilot");
  }

  function handleStreamingChange(isStreamingValue: boolean) {
    setIsStreaming(isStreamingValue);
  }

  return {
    state: {
      greetingName,
      quickActions,
      isLoading: isUserLoading,
      isCreating,
      hasSession,
      initialPrompt,
      isReady: isFlagReady && isChatEnabled !== false && isLoggedIn,
    },
    handlers: {
      handleQuickAction,
      startChatWithPrompt,
      handleSessionNotFound,
      handleStreamingChange,
    },
  };
}
