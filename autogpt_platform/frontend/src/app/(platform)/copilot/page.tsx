"use client";

import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Chat } from "@/components/contextual/Chat/Chat";
import { ChatInput } from "@/components/contextual/Chat/components/ChatInput/ChatInput";
import { getHomepageRoute } from "@/lib/constants";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  Flag,
  type FlagValues,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";
import { useFlags } from "launchdarkly-react-client-sdk";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { getGreetingName, getQuickActions } from "./helpers";

type PageState =
  | { type: "welcome" }
  | { type: "creating"; prompt: string }
  | { type: "chat"; sessionId: string; initialPrompt?: string };

export default function CopilotPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, isLoggedIn, isUserLoading } = useSupabase();

  const isChatEnabled = useGetFlag(Flag.CHAT);
  const flags = useFlags<FlagValues>();
  const homepageRoute = getHomepageRoute(isChatEnabled);
  const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const isLaunchDarklyConfigured = envEnabled && Boolean(clientId);
  const isFlagReady =
    !isLaunchDarklyConfigured || flags[Flag.CHAT] !== undefined;

  const [pageState, setPageState] = useState<PageState>({ type: "welcome" });
  const initialPromptRef = useRef<Map<string, string>>(new Map());

  const urlSessionId = searchParams.get("sessionId");

  // Sync with URL sessionId (preserve initialPrompt from ref)
  useEffect(
    function syncSessionFromUrl() {
      if (urlSessionId) {
        // If we're already in chat state with this sessionId, don't overwrite
        if (pageState.type === "chat" && pageState.sessionId === urlSessionId) {
          return;
        }
        // Get initialPrompt from ref or current state
        const storedInitialPrompt = initialPromptRef.current.get(urlSessionId);
        const currentInitialPrompt =
          storedInitialPrompt ||
          (pageState.type === "creating"
            ? pageState.prompt
            : pageState.type === "chat"
              ? pageState.initialPrompt
              : undefined);
        if (currentInitialPrompt) {
          initialPromptRef.current.set(urlSessionId, currentInitialPrompt);
        }
        setPageState({
          type: "chat",
          sessionId: urlSessionId,
          initialPrompt: currentInitialPrompt,
        });
      } else if (pageState.type === "chat") {
        setPageState({ type: "welcome" });
      }
    },
    [urlSessionId],
  );

  useEffect(
    function ensureAccess() {
      if (!isFlagReady) return;
      if (isChatEnabled === false) {
        router.replace(homepageRoute);
      }
    },
    [homepageRoute, isChatEnabled, isFlagReady, router],
  );

  const greetingName = useMemo(
    function getName() {
      return getGreetingName(user);
    },
    [user],
  );

  const quickActions = useMemo(function getActions() {
    return getQuickActions();
  }, []);

  async function startChatWithPrompt(prompt: string) {
    if (!prompt?.trim()) return;
    if (pageState.type === "creating") return;

    const trimmedPrompt = prompt.trim();
    setPageState({ type: "creating", prompt: trimmedPrompt });

    try {
      // Create session
      const sessionResponse = await postV2CreateSession({
        body: JSON.stringify({}),
      });

      if (sessionResponse.status !== 200 || !sessionResponse.data?.id) {
        throw new Error("Failed to create session");
      }

      const sessionId = sessionResponse.data.id;

      // Store initialPrompt in ref so it persists across re-renders
      initialPromptRef.current.set(sessionId, trimmedPrompt);

      // Update URL and show Chat with initial prompt
      // Chat will handle sending the message and streaming
      window.history.replaceState(null, "", `/copilot?sessionId=${sessionId}`);
      setPageState({ type: "chat", sessionId, initialPrompt: trimmedPrompt });
    } catch (error) {
      console.error("[CopilotPage] Failed to start chat:", error);
      setPageState({ type: "welcome" });
    }
  }

  function handleQuickAction(action: string) {
    startChatWithPrompt(action);
  }

  function handleSessionNotFound() {
    router.replace("/copilot");
  }

  if (!isFlagReady || isChatEnabled === false || !isLoggedIn) {
    return null;
  }

  // Show Chat when we have an active session
  if (pageState.type === "chat") {
    return (
      <div className="flex h-full flex-col">
        <Chat
          key={pageState.sessionId ?? "welcome"}
          className="flex-1"
          urlSessionId={pageState.sessionId}
          initialPrompt={pageState.initialPrompt}
          onSessionNotFound={handleSessionNotFound}
        />
      </div>
    );
  }

  // Show loading state while creating session and sending first message
  if (pageState.type === "creating") {
    return (
      <div className="flex h-full flex-1 flex-col items-center justify-center bg-[#f8f8f9] px-6 py-10">
        <LoadingSpinner size="large" />
        <Text variant="body" className="mt-4 text-zinc-500">
          Starting your chat...
        </Text>
      </div>
    );
  }

  // Show Welcome screen
  const isLoading = isUserLoading;

  return (
    <div className="flex h-full flex-1 items-center justify-center overflow-y-auto bg-[#f8f8f9] px-6 py-10">
      <div className="w-full text-center">
        {isLoading ? (
          <div className="mx-auto max-w-2xl">
            <Skeleton className="mx-auto mb-3 h-8 w-64" />
            <Skeleton className="mx-auto mb-8 h-6 w-80" />
            <div className="mb-8">
              <Skeleton className="mx-auto h-14 w-full rounded-lg" />
            </div>
            <div className="flex flex-wrap items-center justify-center gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-48 rounded-md" />
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="mx-auto max-w-2xl">
              <Text
                variant="h3"
                className="mb-3 !text-[1.375rem] text-zinc-700"
              >
                Hey, <span className="text-violet-600">{greetingName}</span>
              </Text>
              <Text variant="h3" className="mb-8 !font-normal">
                What do you want to automate?
              </Text>

              <div className="mb-6">
                <ChatInput
                  onSend={startChatWithPrompt}
                  placeholder='You can search or just ask - e.g. "create a blog post outline"'
                />
              </div>
            </div>
            <div className="flex flex-nowrap items-center justify-center gap-3 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
              {quickActions.map((action) => (
                <Button
                  key={action}
                  type="button"
                  variant="outline"
                  size="small"
                  onClick={() => handleQuickAction(action)}
                  className="h-auto shrink-0 border-zinc-600 !px-4 !py-2 text-[1rem] text-zinc-600"
                >
                  {action}
                </Button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
