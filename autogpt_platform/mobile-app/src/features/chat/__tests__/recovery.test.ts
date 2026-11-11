import type { UIMessage } from "ai";

import { SignedOutError } from "../../../lib/auth/errors";
import { PLAN_REQUIRED_MESSAGE } from "../chatErrors";
import {
  isRecoverableChatError,
  lastMessageIsUser,
  lastUserText,
  turnLanded,
} from "../recovery";

function message(id: string, role: "user" | "assistant", text: string) {
  return { id, role, parts: [{ type: "text", text }] } satisfies UIMessage;
}

const messages: UIMessage[] = [
  message("m1", "assistant", "Hello"),
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

describe("turnLanded", () => {
  const local = [
    message("l1", "user", "earlier"),
    message("l2", "assistant", "ok"),
    message("l3", "user", "hello"),
  ];

  it("is true while the turn still runs on the server", () => {
    expect(turnLanded([], local, { turn_id: "t1" })).toBe(true);
  });

  it("is true for an answered message with attached files", () => {
    const rows = [
      { role: "user", content: "earlier" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "hello\n\n[Attached files]\n- notes.txt" },
      { role: "assistant", content: "hi" },
    ];

    expect(turnLanded(rows, local, null)).toBe(true);
  });

  it("is false for a repeat that never left the phone", () => {
    const repeat = [
      message("l1", "user", "yes"),
      message("l2", "assistant", "Shall I go on?"),
      message("l3", "user", "yes"),
    ];
    const rows = [
      { role: "user", content: "yes" },
      { role: "assistant", content: "Shall I go on?" },
    ];

    expect(turnLanded(rows, repeat, null)).toBe(false);
  });

  it("is false for a message stored without a reply", () => {
    const rows = [
      { role: "user", content: "earlier" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "hello" },
    ];

    expect(turnLanded(rows, local, null)).toBe(false);
  });

  it("is false when the last server message says something else", () => {
    const rows = [
      { role: "user", content: "earlier" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "something else" },
      { role: "assistant", content: "hi" },
    ];

    expect(turnLanded(rows, local, null)).toBe(false);
  });

  it("is false for no rows and for junk rows", () => {
    expect(turnLanded([], local, null)).toBe(false);
    expect(turnLanded([null, 42, "hello", { role: "user" }], local, null)).toBe(
      false,
    );
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
