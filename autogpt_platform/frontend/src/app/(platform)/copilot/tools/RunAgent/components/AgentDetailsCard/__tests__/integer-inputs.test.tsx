import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { expect, test, vi } from "vitest";
import { AgentDetailsCard } from "../AgentDetailsCard";

const onSend = vi.fn();

vi.mock(
  "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend }),
  }),
);

// The input schema of a workflow whose number input is optional, as
// AgentNumberInputBlock's `Optional[int]` makes it.
function optionalIntegerOutput(bounds = {}): AgentDetailsResponse {
  return {
    type: "agent_details",
    message: "Ready to run.",
    user_authenticated: true,
    agent: {
      id: "g-lookup",
      name: "Order Lookup",
      description: "Looks up an order",
      inputs: {
        type: "object",
        properties: {
          order_id: {
            title: "Order ID",
            anyOf: [{ type: "integer", ...bounds }, { type: "null" }],
            default: null,
          },
        },
      },
      execution_options: { manual: true, scheduled: false, webhook: false },
    },
  } as AgentDetailsResponse;
}

test("runs with an integer beyond 2^53 exactly as it was typed", async () => {
  render(<AgentDetailsCard output={optionalIntegerOutput()} />);

  await expectTypingKept(
    screen.getByRole("spinbutton") as HTMLInputElement,
    "9007199254740993",
  );
  const proceed = screen.getByRole("button", { name: "Proceed" });
  expect(proceed).toHaveProperty("disabled", false);
  fireEvent.click(proceed);

  expect(onSend).toHaveBeenCalledWith(
    'Run the workflow "Order Lookup" with these inputs: {\n  "order_id": "9007199254740993"\n}',
  );
});

test("keeps Proceed off for a big integer below the input's minimum", async () => {
  render(<AgentDetailsCard output={optionalIntegerOutput({ minimum: 0 })} />);
  const field = screen.getByRole("spinbutton") as HTMLInputElement;
  const proceed = screen.getByRole("button", { name: "Proceed" });

  await expectTypingKept(field, "-9007199254740993");
  expect(proceed).toHaveProperty("disabled", true);

  fireEvent.change(field, { target: { value: "" } });
  await expectTypingKept(field, "9007199254740993");
  expect(proceed).toHaveProperty("disabled", false);
});
