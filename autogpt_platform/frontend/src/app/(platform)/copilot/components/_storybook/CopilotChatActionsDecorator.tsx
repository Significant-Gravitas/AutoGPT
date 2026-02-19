import type { Decorator } from "@storybook/nextjs";
import { fn } from "@storybook/test";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";

export const withCopilotChatActions: Decorator = (Story) => (
  <CopilotChatActionsContext.Provider value={{ onSend: fn() }}>
    <Story />
  </CopilotChatActionsContext.Provider>
);
