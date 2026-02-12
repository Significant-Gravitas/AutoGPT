import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useCallback, useEffect, useRef } from "react";
import { convertChatSessionMessagesToUiMessages } from "../helpers/convertChatSessionToUiMessages";

const OPERATING_TYPES = new Set([
  "operation_started",
  "operation_pending",
  "operation_in_progress",
]);

const POLL_INTERVAL_MS = 1_500;

/**
 * Detects whether any message contains a tool part whose output indicates
 * a long-running operation is still in progress.
 */
function hasOperatingTool(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  for (const msg of messages) {
    for (const part of msg.parts) {
      if (!part.type.startsWith("tool-")) continue;
      const toolPart = part as { output?: unknown };
      if (!toolPart.output) continue;
      const output =
        typeof toolPart.output === "string"
          ? safeParse(toolPart.output)
          : toolPart.output;
      if (
        output &&
        typeof output === "object" &&
        "type" in output &&
        OPERATING_TYPES.has((output as { type: string }).type)
      ) {
        return true;
      }
    }
  }
  return false;
}

function safeParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

/**
 * Polls the session endpoint while any tool is in an "operating" state
 * (operation_started / operation_pending / operation_in_progress).
 *
 * When the session data shows the tool output has changed (e.g. to
 * agent_saved), it calls `setMessages` with the updated messages.
 *
 * Fetches session data directly (bypassing the shared React Query cache)
 * so that polling never triggers the hydration effect in useCopilotPage.
 */
export function useLongRunningToolPolling(
  sessionId: string | null,
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  setMessages: (
    updater: (
      prev: UIMessage<unknown, UIDataTypes, UITools>[],
    ) => UIMessage<unknown, UIDataTypes, UITools>[],
  ) => void,
) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    if (!sessionId) return;

    try {
      // Fetch directly instead of refetching the shared session query.
      // Using refetchQueries updated the React Query cache which triggered
      // the hydration effect in useCopilotPage, potentially overwriting
      // useChat's internal message state and breaking subsequent sends.
      const response = await getV2GetSession(sessionId);

      if (response.status !== 200) return;

      const data = response.data as { messages?: unknown[] };
      if (!data.messages) return;

      const freshMessages = convertChatSessionMessagesToUiMessages(
        sessionId,
        data.messages,
      );

      if (!freshMessages || freshMessages.length === 0) return;

      // Update when the long-running tool completed
      if (!hasOperatingTool(freshMessages)) {
        setMessages(() => freshMessages);
        stopPolling();
      }
    } catch {
      // Network error — ignore and retry on next interval
    }
  }, [sessionId, setMessages, stopPolling]);

  useEffect(() => {
    const shouldPoll = hasOperatingTool(messages);

    // Always clear any previous interval first so we never leak timers
    // when the effect re-runs due to dependency changes (e.g. messages
    // updating as the LLM streams text after the tool call).
    stopPolling();

    if (shouldPoll && sessionId) {
      intervalRef.current = setInterval(() => {
        poll();
      }, POLL_INTERVAL_MS);
    }

    return () => {
      stopPolling();
    };
  }, [messages, sessionId, poll, stopPolling]);
}
