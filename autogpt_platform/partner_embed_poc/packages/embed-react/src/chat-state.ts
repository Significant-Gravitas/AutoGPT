import type { UseChatHelpers } from "@ai-sdk/react";
import type { UIMessage } from "ai";

export type EmbeddedChatState = UseChatHelpers<UIMessage> & {
  initializationError: string | null;
  isInitialized: boolean;
};
