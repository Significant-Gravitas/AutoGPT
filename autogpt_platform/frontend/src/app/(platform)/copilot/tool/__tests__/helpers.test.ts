import { describe, expect, it } from "vitest";
import { applyEvent, type ChatMessage, type SampleEvent } from "../helpers";

function apply(messages: ChatMessage[], event: SampleEvent) {
  return applyEvent(messages, event);
}

describe("applyEvent", () => {
  it("adds user and assistant messages", () => {
    const userMessages = apply([], {
      delay: 0,
      kind: "user",
      id: "user-1",
      text: "Hello",
    });
    const messages = apply(userMessages, {
      delay: 0,
      kind: "assistant-start",
      id: "assistant-1",
    });

    expect(messages).toEqual([
      {
        id: "user-1",
        role: "user",
        parts: [{ type: "text", text: "Hello" }],
      },
      { id: "assistant-1", role: "assistant", parts: [] },
    ]);
  });

  it("streams text and reasoning into the matching assistant message", () => {
    let messages = apply([], {
      delay: 0,
      kind: "assistant-start",
      id: "assistant-1",
    });
    messages = apply(messages, {
      delay: 0,
      kind: "text-start",
      messageId: "assistant-1",
    });
    messages = apply(messages, {
      delay: 0,
      kind: "text-delta",
      messageId: "assistant-1",
      delta: "Answer",
    });
    messages = apply(messages, {
      delay: 0,
      kind: "reasoning-start",
      messageId: "assistant-1",
    });
    messages = apply(messages, {
      delay: 0,
      kind: "reasoning-delta",
      messageId: "assistant-1",
      delta: "Thinking",
    });
    messages = apply(messages, {
      delay: 0,
      kind: "reasoning-done",
      messageId: "assistant-1",
    });

    expect(messages[0].parts).toEqual([
      { type: "text", text: "Answer" },
      { type: "reasoning", text: "Thinking", state: "done" },
    ]);
  });

  it("updates only the matching tool call with output or error", () => {
    let messages = apply([], {
      delay: 0,
      kind: "assistant-start",
      id: "assistant-1",
    });
    messages = apply(messages, {
      delay: 0,
      kind: "tool-start",
      messageId: "assistant-1",
      toolCallId: "tool-1",
      tool: "web_search",
      input: { query: "EV sales" },
    });
    messages = apply(messages, {
      delay: 0,
      kind: "tool-start",
      messageId: "assistant-1",
      toolCallId: "tool-2",
      tool: "web_fetch",
      input: { url: "https://example.com" },
    });
    messages = apply(messages, {
      delay: 0,
      kind: "tool-output",
      messageId: "assistant-1",
      toolCallId: "tool-1",
      output: { results: [] },
    });
    messages = apply(messages, {
      delay: 0,
      kind: "tool-error",
      messageId: "assistant-1",
      toolCallId: "tool-2",
      errorText: "Not found",
    });

    expect(messages[0].parts).toEqual([
      expect.objectContaining({
        toolCallId: "tool-1",
        state: "output-available",
        output: { results: [] },
      }),
      expect.objectContaining({
        toolCallId: "tool-2",
        state: "output-error",
        errorText: "Not found",
      }),
    ]);
  });

  it("leaves messages unchanged for control events and missing targets", () => {
    const messages: ChatMessage[] = [
      { id: "assistant-1", role: "assistant", parts: [] },
    ];

    expect(apply(messages, { delay: 0, kind: "await-user" })).toBe(messages);
    expect(
      apply(messages, { delay: 0, kind: "status", message: "Working" }),
    ).toBe(messages);
    expect(
      apply(messages, {
        delay: 0,
        kind: "text-delta",
        messageId: "missing",
        delta: "ignored",
      }),
    ).toEqual(messages);
    expect(
      apply(messages, {
        delay: 0,
        kind: "reasoning-delta",
        messageId: "assistant-1",
        delta: "ignored",
      }),
    ).toEqual(messages);
    expect(
      apply(messages, {
        delay: 0,
        kind: "tool-output",
        messageId: "assistant-1",
        toolCallId: "missing",
        output: null,
      }),
    ).toEqual(messages);
  });
});
