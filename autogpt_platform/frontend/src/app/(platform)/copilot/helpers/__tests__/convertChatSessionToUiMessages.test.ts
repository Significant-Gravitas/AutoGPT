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
});
