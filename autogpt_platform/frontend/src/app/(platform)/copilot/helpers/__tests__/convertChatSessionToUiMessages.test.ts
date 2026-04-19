import { describe, expect, it } from "vitest";
import { convertChatSessionMessagesToUiMessages } from "../convertChatSessionToUiMessages";

const SESSION_ID = "sess-test";

describe("convertChatSessionMessagesToUiMessages", () => {
  it("does not drop user messages with null content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: null, sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("does not drop user messages with empty string content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: "", sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("still drops non-user messages with null content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "assistant", content: null, sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(0);
  });

  it("still drops non-user messages with empty string content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "assistant", content: "", sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(0);
  });

  it("includes user message with normal content", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [{ role: "user", content: "hello", sequence: 0 }],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
  });

  it("attaches a reasoning row between user/assistant to the surrounding assistant bubble", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "user", content: "hi", sequence: 0 },
        { role: "reasoning", content: "thinking deeply", sequence: 1 },
        { role: "assistant", content: "hello back", sequence: 2 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].role).toBe("user");
    const assistant = result.messages[1];
    expect(assistant.role).toBe("assistant");
    expect(assistant.parts).toHaveLength(2);
    expect(assistant.parts[0]).toMatchObject({
      type: "reasoning",
      text: "thinking deeply",
      state: "done",
    });
    expect(assistant.parts[1]).toMatchObject({
      type: "text",
      text: "hello back",
    });
  });

  it("merges separate reasoning and assistant DB rows into one UIMessage with reasoning first", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "plan", sequence: 0 },
        { role: "assistant", content: "answer", sequence: 1 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("assistant");
    expect(result.messages[0].parts.map((p) => p.type)).toEqual([
      "reasoning",
      "text",
    ]);
  });

  it("includes every reasoning row when multiple are present in the same turn", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "step one", sequence: 0 },
        { role: "reasoning", content: "step two", sequence: 1 },
        { role: "assistant", content: "done", sequence: 2 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    const parts = result.messages[0].parts;
    const reasoningParts = parts.filter((p) => p.type === "reasoning");
    expect(reasoningParts).toHaveLength(2);
    expect(reasoningParts[0]).toMatchObject({ text: "step one" });
    expect(reasoningParts[1]).toMatchObject({ text: "step two" });
  });

  it("skips reasoning rows with empty content so no reasoning part is emitted", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "", sequence: 0 },
        { role: "assistant", content: "answer", sequence: 1 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    const parts = result.messages[0].parts;
    expect(parts.some((p) => p.type === "reasoning")).toBe(false);
    expect(parts[0]).toMatchObject({ type: "text", text: "answer" });
  });

  it("captures duration_ms from the following assistant row on the merged bubble", () => {
    const result = convertChatSessionMessagesToUiMessages(
      SESSION_ID,
      [
        { role: "reasoning", content: "ponder", sequence: 0 },
        { role: "assistant", content: "reply", sequence: 1, duration_ms: 750 },
      ],
      { isComplete: true },
    );

    expect(result.messages).toHaveLength(1);
    const mergedId = result.messages[0].id;
    expect(result.durations.get(mergedId)).toBe(750);
  });
});
