import { describe, expect, it } from "vitest";

import { persistedMessagesToUI } from "./session-messages";

describe("persistedMessagesToUI", () => {
  it("restores reasoning and tool output parts from chat rows", () => {
    const messages = persistedMessagesToUI([
      {
        id: "r1",
        role: "reasoning",
        content: "Checking tenant data",
        sequence: 1,
      },
      {
        id: "a1",
        role: "assistant",
        content: "",
        sequence: 2,
        tool_calls: [
          {
            id: "call-1",
            function: {
              name: "query_forwarding_digital",
              arguments: '{"report":"operations_summary"}',
            },
          },
        ],
      },
      {
        id: "t1",
        role: "tool",
        tool_call_id: "call-1",
        content: '{"active_jobs":148}',
        sequence: 3,
      },
    ]);

    expect(messages[0]).toMatchObject({
      role: "assistant",
      parts: [
        { type: "reasoning", text: "Checking tenant data", state: "done" },
      ],
    });
    expect(messages[1].parts[0]).toMatchObject({
      type: "dynamic-tool",
      toolName: "query_forwarding_digital",
      state: "output-available",
      output: { active_jobs: 148 },
    });
  });
});
