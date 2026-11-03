import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { expect, test, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import heldConfirm from "../../../ToolChain/__tests__/heldConfirm.json";
import soulProposal from "../../../ToolChain/__tests__/soulProposal.json";
import { HeldOutcomesContext } from "../../HeldOutcomesContext";
import { answeredLabel, getHeldOutcomes } from "../../heldCallRows";
import { ChainMessageParts } from "../ChainMessageParts";

// Both payloads are what the backend builds: expert_change_gate_test.py.
type Message = UIMessage<unknown, UIDataTypes, UITools>;

const heldTurn: Message = {
  id: "a1",
  role: "assistant",
  parts: [heldConfirm.part as ToolUIPart],
};
const lateResult: Message = {
  id: "u1",
  role: "user",
  metadata: heldConfirm.result.metadata,
  parts: [{ type: "text", text: heldConfirm.result.content }],
};

function renderTurn(parts: Message["parts"], messages: Message[] = []) {
  const onSend = vi.fn();
  const view = render(
    <CopilotChatActionsProvider onSend={onSend}>
      <HeldOutcomesContext.Provider value={getHeldOutcomes(messages)}>
        <ChainMessageParts
          parts={parts}
          messageID="a1"
          isCurrentlyStreaming={false}
        />
      </HeldOutcomesContext.Provider>
    </CopilotChatActionsProvider>,
  );
  return { ...view, onSend };
}

test("a held call drawn outside the chain renders its real card once answered", () => {
  const { rerender } = renderTurn(heldTurn.parts, [heldTurn]);
  expect(screen.queryByText("Expert created")).toBeNull();

  rerender(
    <HeldOutcomesContext.Provider
      value={getHeldOutcomes([heldTurn, lateResult])}
    >
      <ChainMessageParts
        parts={heldTurn.parts}
        messageID="a1"
        isCurrentlyStreaming={false}
      />
    </HeldOutcomesContext.Provider>,
  );

  expect(screen.getByText("Expert created")).toBeDefined();
  expect(screen.getByText("Otto")).toBeDefined();
});

// The card and the wake's label read the same result row; each says what the
// answer did. Bodies are what held._outcome writes for a held action.
test.each([
  ["approved", heldConfirm.result.content, true, "Result sent to Otto."],
  [
    "rejected",
    "Nothing ran: the user declined this action. Do not retry it or reach the same effect another way.",
    false,
    "Declined; nothing was sent to Otto.",
  ],
  [
    "expired",
    "Nothing ran: the approval expired an hour after it was given. Propose the call again if it is still needed.",
    false,
    "Expired; nothing was sent to Otto.",
  ],
  [
    "closed",
    "Nothing ran: this card is no longer open.",
    false,
    "Closed; nothing ran.",
  ],
])(
  "an out-of-chain hire answered %s shows its card only if it ran, and a true label",
  (outcome, body, created, clause) => {
    const messages = answeredHire(outcome, body);
    renderTurn(heldTurn.parts, messages);

    expect(screen.queryByText("Expert created") !== null).toBe(created);
    expect(answeredLabel(messages, 1)).toBe(
      `Approval answered: Confirm the team change. ${clause}`,
    );
  },
);

test("a Soul edit is put to the user on its own card, whose Approve the gate takes", async () => {
  const user = userEvent.setup();
  const { onSend } = renderTurn([
    {
      type: "tool-update_expert_soul",
      state: "output-available",
      toolCallId: "call-1",
      input: { voice_preferences: "Warmer." },
      output: soulProposal,
    } as ToolUIPart,
  ]);

  expect(screen.getByText("Update Soul")).toBeDefined();
  expect(screen.getByText("Voice")).toBeDefined();
  expect(screen.getByText("Warmer.")).toBeDefined();
  expect(screen.getByText("Was: Short and factual.")).toBeDefined();

  await user.click(screen.getByRole("button", { name: "Approve" }));

  // The line gate/card_approval.py takes for confirm_expert_soul_update.
  expect(onSend).toHaveBeenCalledWith(
    "Approved: update your Soul (confirmation_id: 5f0c2a3e-7b1d-4c9e-8a6f-0d2b4e6c8a10).",
  );
});

function answeredHire(outcome: string, body: string): Message[] {
  const wake: Message = {
    id: "w1",
    role: "user",
    metadata: { held_calls_answered: true },
    parts: [
      {
        type: "text",
        text: "I answered an action that was waiting for my approval.",
      },
    ],
  };
  const header = heldConfirm.result.content.split("\n", 1)[0];
  const result: Message = {
    id: "u1",
    role: "user",
    metadata: {
      held_call: { ...heldConfirm.result.metadata.held_call, outcome },
    },
    parts: [
      {
        type: "text",
        text: body.startsWith("<held_call_result")
          ? body
          : `${header}\n${body}\n</held_call_result>`,
      },
    ],
  };
  return [heldTurn, wake, result];
}
