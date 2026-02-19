import type { Decorator } from "@storybook/nextjs";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";

export const withCopilotChatActions: Decorator = (Story) => (
  <CopilotChatActionsContext.Provider
    value={{ onSend: (msg) => console.log("[Storybook] onSend:", msg) }}
  >
    <Story />
  </CopilotChatActionsContext.Provider>
);
