import { getGetV2GetSessionQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
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
  const queryClient = useQueryClient();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    if (!sessionId) return;

    // Invalidate the query cache so the next fetch gets fresh data
    await queryClient.invalidateQueries({
      queryKey: getGetV2GetSessionQueryKey(sessionId),
    });

    // Fetch fresh session data
    const data = queryClient.getQueryData<{
      status: number;
      data: { messages?: unknown[] };
    }>(getGetV2GetSessionQueryKey(sessionId));

    if (data?.status !== 200 || !data.data.messages) return;

    const freshMessages = convertChatSessionMessagesToUiMessages(
      sessionId,
      data.data.messages,
    );

    if (!freshMessages || freshMessages.length === 0) return;

    // Update when the long-running tool completed
    if (!hasOperatingTool(freshMessages)) {
      setMessages(() => freshMessages);
      stopPolling();
    }
  }, [sessionId, queryClient, setMessages, stopPolling]);

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
