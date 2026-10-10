import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { campaign } from "@/lib/openui/__tests__/sample-fixtures";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";
import type { RenderUIMessagePart } from "../isRenderUIPart";

const message = "View not published: Duplicate definition: note.";

describe("OpenUI repair feedback", () => {
  it.each(["direct", "deferred", "dynamic", "transport"])(
    "keeps a rejected %s attempt in tool history alongside its corrected view",
    async (transport) => {
      const rejected: RenderUIMessagePart = {
        type: "tool-render_ui",
        toolCallId: "rejected",
        state: "output-available",
        input: {},
        output: { type: "error", message },
      };
      if (transport === "deferred") {
        rejected.type = "tool-run_capability";
        rejected.input = { id: "tool:render_ui", input: {} };
        rejected.output = JSON.stringify(rejected.output);
      }
      const part: RenderUIMessagePart =
        transport === "dynamic"
          ? { ...rejected, type: "dynamic-tool", toolName: "render_ui" }
          : transport === "transport"
            ? {
                type: "tool-render_ui",
                toolCallId: "rejected",
                state: "output-error",
                input: {},
                errorText: message,
              }
            : rejected;
      render(
        <CopilotChatActionsProvider onSend={vi.fn()}>
          <ChainMessageParts
            parts={[
              part,
              {
                type: "tool-render_ui",
                toolCallId: "corrected",
                state: "output-available",
                input: {},
                output: {
                  type: "ui_rendered",
                  version: 1,
                  source: campaign,
                  message: "A corrected campaign brief.",
                },
              },
            ]}
            messageID="repaired-message"
            isCurrentlyStreaming={false}
          />
        </CopilotChatActionsProvider>,
      );
      expect(await screen.findByLabelText("Audience")).toBeDefined();
      expect(
        screen.queryByRole("button", { name: "Rebuild this view" }),
      ).toBeNull();
      expect(screen.queryByText(/Here’s the saved summary/)).toBeNull();
      expect(screen.getByText(message)).toBeDefined();
    },
  );
});
