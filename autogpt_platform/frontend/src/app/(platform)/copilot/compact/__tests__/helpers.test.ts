import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import { buildCompactBlocks, deriveAgentStatus } from "../helpers";

function message(
  role: "user" | "assistant",
  parts: UIMessage["parts"],
): UIMessage {
  return { id: `${role}-${parts.length}`, role, parts };
}

const userAsk = message("user", [{ type: "text", text: "Find hotels" }]);

const runningSearch = {
  type: "tool-web_search",
  toolCallId: "call-1",
  state: "input-available",
  input: { query: "hotels" },
} as unknown as UIMessage["parts"][number];

const finishedSearch = {
  ...runningSearch,
  state: "output-available",
  output: { results: [] },
} as unknown as UIMessage["parts"][number];

const question = {
  type: "tool-ask_question",
  toolCallId: "call-2",
  state: "output-available",
  input: {},
  output: { type: "x", message: "Which city?", questions: [] },
} as unknown as UIMessage["parts"][number];

describe("deriveAgentStatus", () => {
  it("is idle with no messages", () => {
    expect(
      deriveAgentStatus({
        status: "ready",
        messages: [],
        hasPendingReviews: false,
      }),
    ).toBe("idle");
  });

  it("is thinking right after a send", () => {
    expect(
      deriveAgentStatus({
        status: "submitted",
        messages: [userAsk],
        hasPendingReviews: false,
      }),
    ).toBe("thinking");
  });

  it("is working while a tool call runs", () => {
    expect(
      deriveAgentStatus({
        status: "streaming",
        messages: [userAsk, message("assistant", [runningSearch])],
        hasPendingReviews: false,
      }),
    ).toBe("working");
  });

  it("is waiting when a review is pending, even mid-stream", () => {
    expect(
      deriveAgentStatus({
        status: "streaming",
        messages: [userAsk, message("assistant", [runningSearch])],
        hasPendingReviews: true,
      }),
    ).toBe("waiting");
  });

  it("is waiting when the turn ended on a question", () => {
    expect(
      deriveAgentStatus({
        status: "ready",
        messages: [userAsk, message("assistant", [question])],
        hasPendingReviews: false,
      }),
    ).toBe("waiting");
  });

  it("is done once the reply has settled", () => {
    expect(
      deriveAgentStatus({
        status: "ready",
        messages: [
          userAsk,
          message("assistant", [
            finishedSearch,
            { type: "text", text: "Here are three." },
          ]),
        ],
        hasPendingReviews: false,
      }),
    ).toBe("done");
  });
});

describe("buildCompactBlocks", () => {
  it("folds tool calls into one activity block and keeps prose and cards", () => {
    const blocks = buildCompactBlocks([
      { type: "step-start" },
      { type: "reasoning", text: "hmm", state: "done" },
      runningSearch,
      { ...finishedSearch, toolCallId: "call-3" } as UIMessage["parts"][number],
      { type: "text", text: "Here are three." },
      question,
    ]);

    expect(blocks.map((block) => block.kind)).toEqual([
      "activity",
      "text",
      "card",
    ]);
  });
});
