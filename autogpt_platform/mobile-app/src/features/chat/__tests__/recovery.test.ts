import type { UIMessage } from "ai";

import { SignedOutError } from "@/lib/auth/token";

import {
  isRecoverableChatError,
  lastMessageIsUser,
  lastUserText,
  serverHasUserMessage,
} from "../recovery";
import { PLAN_REQUIRED_MESSAGE } from "../transport";

const messages: UIMessage[] = [
  {
    id: "m1",
    role: "assistant",
    parts: [{ type: "text", text: "Hello" }],
  },
  {
    id: "m2",
    role: "user",
    parts: [
      { type: "text", text: "Plan my " },
      { type: "text", text: "week" },
    ],
  },
];

describe("isRecoverableChatError", () => {
  it("does not recover a dead session", () => {
    expect(isRecoverableChatError(new SignedOutError())).toBe(false);
  });

  it("does not recover a refused request", () => {
    expect(isRecoverableChatError(new Error(PLAN_REQUIRED_MESSAGE))).toBe(
      false,
    );
  });

  it("recovers a network drop", () => {
    expect(isRecoverableChatError(new Error("Network request failed"))).toBe(
      true,
    );
  });

  it("recovers a broken stream", () => {
    expect(isRecoverableChatError(new TypeError("stream closed"))).toBe(true);
  });
});

describe("serverHasUserMessage", () => {
  it("finds the last user row even with attached files after it", () => {
    const rows = [
      { role: "user", content: "earlier" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "hello\n\n[Attached files]\n- notes.txt" },
      { role: "assistant", content: "hi" },
    ];

    expect(serverHasUserMessage(rows, "hello")).toBe(true);
  });

  it("is false when the last user row says something else", () => {
    const rows = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
      { role: "user", content: "something else" },
    ];

    expect(serverHasUserMessage(rows, "hello")).toBe(false);
  });

  it("is false for no rows", () => {
    expect(serverHasUserMessage([], "hello")).toBe(false);
  });

  it("is false for junk rows", () => {
    const rows = [null, 42, "hello", { role: "user" }, { content: "hello" }];

    expect(serverHasUserMessage(rows, "hello")).toBe(false);
  });
});

describe("lastMessageIsUser", () => {
  it("is true when the user spoke last", () => {
    expect(lastMessageIsUser(messages)).toBe(true);
  });

  it("is false when the assistant spoke last", () => {
    expect(lastMessageIsUser([...messages].reverse())).toBe(false);
  });
});

describe("lastUserText", () => {
  it("joins the text parts of the last user message", () => {
    expect(lastUserText(messages)).toBe("Plan my week");
  });

  it("is null without a user message", () => {
    expect(lastUserText(messages.slice(0, 1))).toBeNull();
  });
});
