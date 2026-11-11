import type { UIMessage } from "ai";

import { SignedOutError } from "../../../lib/auth/errors";
import { PLAN_REQUIRED_MESSAGE } from "../chatErrors";
import {
  isRecoverableChatError,
  lastMessageIsUser,
  lastUserText,
  messagesInWindow,
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

  it("is true when older pages loaded here hold more user messages", () => {
    const loaded = [
      message("s1-1", "user", "first"),
      message("s1-2", "assistant", "sure"),
      message("s1-3", "user", "earlier"),
      message("s1-4", "assistant", "ok"),
      message("l3", "user", "hello"),
    ];
    const newestPage = [
      { role: "user", content: "earlier", sequence: 3 },
      { role: "assistant", content: "ok", sequence: 4 },
      { role: "user", content: "hello", sequence: 5 },
      { role: "assistant", content: "hi", sequence: 6 },
    ];

    expect(turnLanded(newestPage, loaded, null)).toBe(false);
    expect(
      turnLanded(newestPage, messagesInWindow("s1", loaded, 3), null),
    ).toBe(true);
  });

  it("is false for no rows and for junk rows", () => {
    expect(turnLanded([], local, null)).toBe(false);
    expect(turnLanded([null, 42, "hello", { role: "user" }], local, null)).toBe(
      false,
    );
  });
});

describe("messagesInWindow", () => {
  const mixed = [
    message("s1-2", "user", "old"),
    message("s1-idx-0", "user", "no sequence"),
    message("s1-7", "assistant", "in window"),
    message("aB3dE5fG7hJ9kL1m", "user", "sent here"),
    message("s2-1", "user", "other session"),
  ];

  it("keeps streamed, sequence-less and in-window messages", () => {
    expect(messagesInWindow("s1", mixed, 5).map((m) => m.id)).toEqual([
      "s1-idx-0",
      "s1-7",
      "aB3dE5fG7hJ9kL1m",
      "s2-1",
    ]);
  });

  it("keeps everything without a window", () => {
    expect(messagesInWindow("s1", mixed, null)).toEqual(mixed);
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
