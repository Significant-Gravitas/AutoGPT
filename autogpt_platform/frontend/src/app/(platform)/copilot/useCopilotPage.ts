import {
  getGetV2ListSessionsQueryKey,
  postV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { SessionKey, sessionStorage } from "@/services/storage/session-storage";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useCopilotStore } from "./copilot-page-store";
import { getGreetingName, getQuickActions } from "./helpers";
import { useCopilotSessionId } from "./useCopilotSessionId";

export function useCopilotPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const { toast } = useToast();
  const { completeStep } = useOnboarding();

  const { urlSessionId, setUrlSessionId } = useCopilotSessionId();
  const setIsStreaming = useCopilotStore((s) => s.setIsStreaming);
  const isCreating = useCopilotStore((s) => s.isCreatingSession);
  const setIsCreating = useCopilotStore((s) => s.setIsCreatingSession);

  const greetingName = getGreetingName(user);
  const quickActions = getQuickActions();

  const hasSession = Boolean(urlSessionId);
  const initialPrompt = urlSessionId
    ? getInitialPrompt(urlSessionId)
    : undefined;

  useEffect(() => {
    if (isLoggedIn) completeStep("VISIT_COPILOT");
  }, [completeStep, isLoggedIn]);

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
      setInitialPrompt(sessionId, trimmedPrompt);

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });

      await setUrlSessionId(sessionId, { shallow: true });
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
      hasSession,
      initialPrompt,
    },
    handlers: {
      handleQuickAction,
      startChatWithPrompt,
      handleSessionNotFound,
      handleStreamingChange,
    },
  };
}

function getInitialPrompt(sessionId: string): string | undefined {
  try {
    const prompts = JSON.parse(
      sessionStorage.get(SessionKey.CHAT_INITIAL_PROMPTS) || "{}",
    );
    return prompts[sessionId];
  } catch {
    return undefined;
  }
}

function setInitialPrompt(sessionId: string, prompt: string): void {
  try {
    const prompts = JSON.parse(
      sessionStorage.get(SessionKey.CHAT_INITIAL_PROMPTS) || "{}",
    );
    prompts[sessionId] = prompt;
    sessionStorage.set(
      SessionKey.CHAT_INITIAL_PROMPTS,
      JSON.stringify(prompts),
    );
  } catch {
    // Ignore storage errors
  }
}
