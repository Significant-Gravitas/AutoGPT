import type { getV2GetSessionResponse } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDataTypes, UIMessage, UITools } from "ai";

import { convertChatSessionMessagesToUiMessages } from "./convertChatSessionToUiMessages";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

/**
 * What a chat panel streaming through `useCopilotStream` reads from its
 * session: the rows to hydrate, whether a turn is running, the session's
 * `chat_status` (a turn queued at the running cap is followed and badged
 * "Queued" from it) and the per-row stats the badge anchors on.
 */
export function readSessionForStream(
  sessionId: string | null,
  response: getV2GetSessionResponse | undefined,
) {
  if (response?.status !== 200 || !sessionId) {
    return {
      hydratedMessages: undefined,
      hasActiveStream: false,
      sessionChatStatus: undefined,
      turnStats: undefined,
    };
  }
  const session = response.data;
  const hasActiveStream = !!session.active_stream;
  const converted = convertChatSessionMessagesToUiMessages(
    sessionId,
    session.messages ?? [],
    { isComplete: !hasActiveStream },
  );
  return {
    hydratedMessages: converted.messages as UiMessages,
    hasActiveStream,
    sessionChatStatus: session.chat_status,
    turnStats: converted.stats,
  };
}
