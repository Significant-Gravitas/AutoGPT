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
import { useEffect, useReducer } from "react";
import { useNewChat } from "./NewChatContext";
import { getGreetingName, getQuickActions, type PageState } from "./helpers";
import { useCopilotURLState } from "./useCopilotURLState";

type CopilotState = {
  pageState: PageState;
  isStreaming: boolean;
  isNewChatModalOpen: boolean;
  initialPrompts: Record<string, string>;
  previousSessionId: string | null;
};

type CopilotAction =
  | { type: "setPageState"; pageState: PageState }
  | { type: "setStreaming"; isStreaming: boolean }
  | { type: "setNewChatModalOpen"; isOpen: boolean }
  | { type: "setInitialPrompt"; sessionId: string; prompt: string }
  | { type: "setPreviousSessionId"; sessionId: string | null };

function isSamePageState(next: PageState, current: PageState) {
  if (next.type !== current.type) return false;
  if (next.type === "creating" && current.type === "creating") {
    return next.prompt === current.prompt;
  }
  if (next.type === "chat" && current.type === "chat") {
    return (
      next.sessionId === current.sessionId &&
      next.initialPrompt === current.initialPrompt
    );
  }
  return true;
}

function copilotReducer(
  state: CopilotState,
  action: CopilotAction,
): CopilotState {
  if (action.type === "setPageState") {
    if (isSamePageState(action.pageState, state.pageState)) return state;
    return { ...state, pageState: action.pageState };
  }
  if (action.type === "setStreaming") {
    if (action.isStreaming === state.isStreaming) return state;
    return { ...state, isStreaming: action.isStreaming };
  }
  if (action.type === "setNewChatModalOpen") {
    if (action.isOpen === state.isNewChatModalOpen) return state;
    return { ...state, isNewChatModalOpen: action.isOpen };
  }
  if (action.type === "setInitialPrompt") {
    if (state.initialPrompts[action.sessionId] === action.prompt) return state;
    return {
      ...state,
      initialPrompts: {
        ...state.initialPrompts,
        [action.sessionId]: action.prompt,
      },
    };
  }
  if (action.type === "setPreviousSessionId") {
    if (state.previousSessionId === action.sessionId) return state;
    return { ...state, previousSessionId: action.sessionId };
  }
  return state;
}

export function useCopilotPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const { toast } = useToast();

  const isChatEnabled = useGetFlag(Flag.CHAT);
  const flags = useFlags<FlagValues>();
  const homepageRoute = getHomepageRoute(isChatEnabled);
  const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const isLaunchDarklyConfigured = envEnabled && Boolean(clientId);
  const isFlagReady =
    !isLaunchDarklyConfigured || flags[Flag.CHAT] !== undefined;

  const [state, dispatch] = useReducer(copilotReducer, {
    pageState: { type: "welcome" },
    isStreaming: false,
    isNewChatModalOpen: false,
    initialPrompts: {},
    previousSessionId: null,
  });

  const newChatContext = useNewChat();
  const greetingName = getGreetingName(user);
  const quickActions = getQuickActions();

  function setPageState(pageState: PageState) {
    dispatch({ type: "setPageState", pageState });
  }

  function setInitialPrompt(sessionId: string, prompt: string) {
    dispatch({ type: "setInitialPrompt", sessionId, prompt });
  }

  function setPreviousSessionId(sessionId: string | null) {
    dispatch({ type: "setPreviousSessionId", sessionId });
  }

  const { setUrlSessionId } = useCopilotURLState({
    pageState: state.pageState,
    initialPrompts: state.initialPrompts,
    previousSessionId: state.previousSessionId,
    setPageState,
    setInitialPrompt,
    setPreviousSessionId,
  });

  useEffect(
    function registerNewChatHandler() {
      if (!newChatContext) return;
      newChatContext.setOnNewChatClick(handleNewChatClick);
      return function cleanup() {
        newChatContext.setOnNewChatClick(undefined);
      };
    },
    [newChatContext, handleNewChatClick],
  );

  useEffect(
    function transitionNewChatToWelcome() {
      if (state.pageState.type === "newChat") {
        function setWelcomeState() {
          dispatch({ type: "setPageState", pageState: { type: "welcome" } });
        }

        const timer = setTimeout(setWelcomeState, 300);

        return function cleanup() {
          clearTimeout(timer);
        };
      }
    },
    [state.pageState.type],
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

  async function startChatWithPrompt(prompt: string) {
    if (!prompt?.trim()) return;
    if (state.pageState.type === "creating") return;

    const trimmedPrompt = prompt.trim();
    dispatch({
      type: "setPageState",
      pageState: { type: "creating", prompt: trimmedPrompt },
    });

    try {
      const sessionResponse = await postV2CreateSession({
        body: JSON.stringify({}),
      });

      if (sessionResponse.status !== 200 || !sessionResponse.data?.id) {
        throw new Error("Failed to create session");
      }

      const sessionId = sessionResponse.data.id;

      dispatch({
        type: "setInitialPrompt",
        sessionId,
        prompt: trimmedPrompt,
      });

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });

      await setUrlSessionId(sessionId, { shallow: false });
      dispatch({
        type: "setPageState",
        pageState: { type: "chat", sessionId, initialPrompt: trimmedPrompt },
      });
    } catch (error) {
      console.error("[CopilotPage] Failed to start chat:", error);
      toast({ title: "Failed to start chat", variant: "destructive" });
      Sentry.captureException(error);
      dispatch({ type: "setPageState", pageState: { type: "welcome" } });
    }
  }

  function handleQuickAction(action: string) {
    startChatWithPrompt(action);
  }

  function handleSessionNotFound() {
    router.replace("/copilot");
  }

  function handleStreamingChange(isStreamingValue: boolean) {
    dispatch({ type: "setStreaming", isStreaming: isStreamingValue });
  }

  async function proceedWithNewChat() {
    dispatch({ type: "setNewChatModalOpen", isOpen: false });
    if (newChatContext?.performNewChat) {
      newChatContext.performNewChat();
      return;
    }
    try {
      await setUrlSessionId(null, { shallow: false });
    } catch (error) {
      console.error("[CopilotPage] Failed to clear session:", error);
    }
    router.replace("/copilot");
  }

  function handleCancelNewChat() {
    dispatch({ type: "setNewChatModalOpen", isOpen: false });
  }

  function handleNewChatModalOpen(isOpen: boolean) {
    dispatch({ type: "setNewChatModalOpen", isOpen });
  }

  function handleNewChatClick() {
    if (state.isStreaming) {
      dispatch({ type: "setNewChatModalOpen", isOpen: true });
    } else {
      proceedWithNewChat();
    }
  }

  return {
    state: {
      greetingName,
      quickActions,
      isLoading: isUserLoading,
      pageState: state.pageState,
      isNewChatModalOpen: state.isNewChatModalOpen,
      isReady: isFlagReady && isChatEnabled !== false && isLoggedIn,
    },
    handlers: {
      handleQuickAction,
      startChatWithPrompt,
      handleSessionNotFound,
      handleStreamingChange,
      handleCancelNewChat,
      proceedWithNewChat,
      handleNewChatModalOpen,
    },
  };
}
