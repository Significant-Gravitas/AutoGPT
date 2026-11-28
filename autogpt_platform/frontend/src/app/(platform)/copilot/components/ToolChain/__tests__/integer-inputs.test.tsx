import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { ToolChain } from "../ToolChain";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false, user: null }),
}));

/** A block that needs one integer input and no account. */
function integerInputPart(): MessagePart {
  return {
    type: "tool-run_block",
    toolCallId: "call-run_block",
    state: "output-available",
    input: { block_name: "OrderLookupBlock" },
    output: {
      type: "setup_requirements",
      message: "Fill in the order to look up.",
      session_id: "session-1",
      setup_info: {
        agent_id: "block-1",
        agent_name: "Order Lookup",
        requirements: {
          inputs: [
            {
              name: "order_id",
              title: "Order ID",
              type: "integer",
              required: true,
            },
          ],
        },
        user_readiness: { has_all_credentials: true, missing_credentials: {} },
      },
    },
  } as unknown as MessagePart;
}

describe("chain input card", () => {
  it("sends an integer beyond 2^53 exactly as it was typed", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[integerInputPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await expectTypingKept(
      (await screen.findByRole("spinbutton")) as HTMLInputElement,
      "9007199254740993",
    );
    fireEvent.click(screen.getByRole("button", { name: "Proceed" }));

    await waitFor(() =>
      expect(onSend).toHaveBeenCalledWith(
        expect.stringContaining('"order_id": "9007199254740993"'),
      ),
    );
  });
});
