import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import { getAppliedExpertConfirmationIDs } from "../ExpertConfirmationContext";

describe("getAppliedExpertConfirmationIDs", () => {
  it("collects successful single and batch confirmations", () => {
    const messages = [
      messageWithOutput({
        type: "expert_change_applied",
        applied: true,
        confirmation_id: "single",
      }),
      messageWithOutput({
        type: "expert_change_batch_applied",
        results: [
          { confirmation_id: "applied", outcome: "applied" },
          { confirmation_id: "retried", outcome: "already_applied" },
          { confirmation_id: "failed", outcome: "failed" },
        ],
      }),
    ];

    expect([...getAppliedExpertConfirmationIDs(messages)]).toEqual([
      "single",
      "applied",
      "retried",
    ]);
  });

  it("ignores malformed and failed outputs", () => {
    const messages = [
      messageWithOutput("not json"),
      messageWithOutput({
        type: "expert_change_applied",
        applied: false,
        confirmation_id: "not-applied",
      }),
    ];

    expect(getAppliedExpertConfirmationIDs(messages).size).toBe(0);
  });
});

function messageWithOutput(output: unknown): UIMessage {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    parts: [
      {
        type: "tool-confirm_expert_change",
        state: "output-available",
        toolCallId: crypto.randomUUID(),
        input: {},
        output,
      },
    ],
  } as UIMessage;
}
