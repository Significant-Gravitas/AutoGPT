import { parseAsString, useQueryState } from "nuqs";
import { useLayoutEffect } from "react";
import {
  getInitialPromptFromState,
  type PageState,
  shouldResetToWelcome,
} from "./helpers";

interface UseCopilotUrlStateArgs {
  pageState: PageState;
  initialPrompts: Record<string, string>;
  previousSessionId: string | null;
  setPageState: (pageState: PageState) => void;
  setInitialPrompt: (sessionId: string, prompt: string) => void;
  setPreviousSessionId: (sessionId: string | null) => void;
}

export function useCopilotURLState({
  pageState,
  initialPrompts,
  previousSessionId,
  setPageState,
  setInitialPrompt,
  setPreviousSessionId,
}: UseCopilotUrlStateArgs) {
  const [urlSessionId, setUrlSessionId] = useQueryState(
    "sessionId",
    parseAsString,
  );

  function syncSessionFromUrl() {
    if (urlSessionId) {
      if (pageState.type === "chat" && pageState.sessionId === urlSessionId) {
        setPreviousSessionId(urlSessionId);
        return;
      }

      const storedInitialPrompt = initialPrompts[urlSessionId];
      const currentInitialPrompt = getInitialPromptFromState(
        pageState,
        storedInitialPrompt,
      );

      if (currentInitialPrompt) {
        setInitialPrompt(urlSessionId, currentInitialPrompt);
      }

      setPageState({
        type: "chat",
        sessionId: urlSessionId,
        initialPrompt: currentInitialPrompt,
      });
      setPreviousSessionId(urlSessionId);
      return;
    }

    const wasInChat = previousSessionId !== null && pageState.type === "chat";
    setPreviousSessionId(null);
    if (wasInChat) {
      setPageState({ type: "newChat" });
      return;
    }

    if (shouldResetToWelcome(pageState)) {
      setPageState({ type: "welcome" });
    }
  }

  useLayoutEffect(syncSessionFromUrl, [
    urlSessionId,
    pageState.type,
    previousSessionId,
    initialPrompts,
  ]);

  return {
    urlSessionId,
    setUrlSessionId,
  };
}
