import { StrictMode } from "react";
import { vi } from "vitest";
import type { ToolUIPart } from "ai";
import { render, screen } from "@/tests/integrations/test-utils";
import { connected } from "@/lib/openui/__tests__/connected-fixtures";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";

export async function showResult(
  send = vi.fn(),
  readOnly = false,
  source = connected,
) {
  const part: ToolUIPart = {
    type: "tool-render_ui",
    toolCallId: "connected-ui",
    state: "output-available",
    input: {},
    output: {
      type: "ui_rendered",
      version: 1,
      session_id: "connected-session",
      source,
      message: "Weekend options",
    },
  };
  const result = render(
    <StrictMode>
      <CopilotChatActionsProvider
        onSend={send}
        chatSurface={readOnly ? "share" : "copilot"}
      >
        <ChainMessageParts
          parts={[part]}
          messageID="connected-message"
          isCurrentlyStreaming={false}
          readOnly={readOnly}
        />
      </CopilotChatActionsProvider>
    </StrictMode>,
  );
  await screen.findByRole(
    "region",
    { name: "Interactive response" },
    { timeout: 10000 },
  );
  return result;
}
