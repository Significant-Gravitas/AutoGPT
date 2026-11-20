import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { v5 as uuidv5 } from "uuid";
import { describe, expect, it } from "vitest";

import { emptyTurnLog, type LogRow, type PersistedRow } from "../turnLog";
import {
  locatePrompt,
  mergeRows,
  newTurnSegment,
  persistedPromptId,
  turnRows,
  userSegment,
  type Segment,
  lastReplySequence,
} from "../turnTail";

describe("persistedPromptId", () => {
  it("is the id the backend's scoped_client_message_id gives a send's row", () => {
    // Pins the id scheme to backend/copilot/expert_kickoff.py's scoped_client_message_id.
    const source = readFileSync(
      resolve(
        dirname(fileURLToPath(import.meta.url)),
        "../../../../../../../backend/backend/copilot/expert_kickoff.py",
      ),
      "utf8",
    );
    const namespace =
      /_CLIENT_MESSAGE_NAMESPACE = uuid\.UUID\("([0-9a-f-]{36})"\)/.exec(
        source,
      )?.[1];
    expect(
      source,
      "scoped_client_message_id's uuid5 name changed: make persistedPromptId build the same one",
    ).toMatch(
      /uuid\.uuid5\(\s*_CLIENT_MESSAGE_NAMESPACE,\s*f"\{user_id\}:\{session_id\}:\{client_message_id\}",?\s*\)/,
    );
    expect(namespace).toBeDefined();
    expect(persistedPromptId("user-a", "session-a", "click-a")).toBe(
      uuidv5("user-a:session-a:click-a", namespace!),
    );
    // uuid.uuid5(_CLIENT_MESSAGE_NAMESPACE, "user-a:session-a:click-a") in Python.
    expect(persistedPromptId("user-a", "session-a", "click-a")).toBe(
      "540dc830-2fef-5bde-bf7c-d61d574b097f",
    );
  });
});

describe("locatePrompt", () => {
  const ROWS: PersistedRow[] = [
    { id: "other-tab", role: "user", content: "hi there", sequence: 4 },
    { id: "reply", role: "assistant", content: "Hello", sequence: 5 },
    { id: "own", role: "user", content: "hi", sequence: 6 },
  ];

  it("finds a send's row by its id, not another prompt that starts with its words", () => {
    const sent = userSegment(prompt("hi"), "sent", "own");
    expect(locatePrompt(sent, ROWS, 4)?.sequence).toBe(6);
  });

  it("finds nothing for a send the session has no row for", () => {
    const sent = userSegment(prompt("hi"), "sent", "never-written");
    expect(locatePrompt(sent, ROWS, 4)).toBeNull();
    const filesOnly = userSegment(prompt(""), "sent", "never-written");
    expect(locatePrompt(filesOnly, ROWS, 4)).toBeNull();
  });

  it("finds a promoted follow-up, which no send names, by its words", () => {
    expect(
      locatePrompt(userSegment(prompt("hi there"), "chip"), ROWS, 4)?.sequence,
    ).toBe(4);
  });
});

describe("turnRows", () => {
  it("stops a stopped turn's rows at the next prompt, before that prompt has a sequence", () => {
    // A send right after a stop: the next prompt is still local when the
    // stopped turn reconciles, and its row is already persisted.
    const stopped = {
      ...newTurnSegment("turn-1", emptyTurnLog()),
      stopped: true,
    };
    const segments: Segment[] = [
      { ...userSegment(prompt("first"), "sent"), sequence: 0 },
      stopped,
      userSegment(prompt("second"), "sent"),
    ];
    const rows: PersistedRow[] = [
      { role: "user", content: "first", sequence: 0 },
      { role: "assistant", content: "Partial", sequence: 1 },
      {
        role: "assistant",
        content: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
        sequence: 2,
      },
      { role: "user", content: "second", sequence: 3 },
      { role: "assistant", content: "Next reply", sequence: 4 },
    ];

    const persisted = turnRows(segments, 0, stopped, rows);

    expect(persisted?.start).toBe(1);
    expect(persisted?.rows.map((r) => r.content)).toEqual([
      "Partial",
      "[__COPILOT_ERROR_f7a1__] Operation cancelled",
    ]);
  });
});

describe("turnRows", () => {
  it("finds a repeated prompt's row after the rows the tail already holds", () => {
    const answered = newTurnSegment("turn-1", {
      ...emptyTurnLog(),
      rows: [row("block-1", "Hello", 1)],
    });
    // Stopped before its first checkpoint, so its start is found by its prompt.
    const stopped = {
      ...newTurnSegment("turn-2", emptyTurnLog()),
      stopped: true,
    };
    const segments: Segment[] = [
      { ...userSegment(prompt("hi"), "sent"), sequence: 0 },
      answered,
      userSegment(prompt("hi"), "sent"),
      stopped,
    ];
    const rows: PersistedRow[] = [
      { role: "user", content: "hi", sequence: 0 },
      { role: "assistant", content: "Hello", sequence: 1 },
      { role: "user", content: "hi", sequence: 2 },
      { role: "assistant", content: "Partial", sequence: 3 },
    ];

    const persisted = turnRows(segments, 0, stopped, rows);

    expect(persisted?.start).toBe(3);
    expect(persisted?.rows.map((r) => r.content)).toEqual(["Partial"]);
  });
});

describe("mergeRows", () => {
  it("keeps every on-screen key, and an equal row's object", () => {
    const live = [row("block-1", "Hello", 1), row("block-2", "Bye", null)];
    const persisted = [row("seq:1", "Hello", 1), row("seq:2", "Goodbye", 2)];

    const merged = mergeRows(live, persisted);

    expect(merged.map((r) => r.key)).toEqual(["block-1", "block-2"]);
    expect(merged[0]).toBe(live[0]);
    expect(merged[1]).toMatchObject({ content: "Goodbye", sequence: 2 });
  });

  it("drops streamed rows past the DB's end once an earlier row is out of place", () => {
    // The fold holds one row the DB does not, so every later index is shifted.
    const live = [
      row("block-0", "Thinking", null),
      row("block-1", "Hello", null),
      row("block-2", "Bye", null),
    ];
    const persisted = [row("seq:1", "Hello", 1), row("seq:2", "Bye", 2)];

    expect(mergeRows(live, persisted).map((r) => r.content)).toEqual([
      "Hello",
      "Bye",
    ]);
  });
});

function row(key: string, content: string, sequence: number | null): LogRow {
  return {
    key,
    role: "assistant",
    content,
    toolCalls: [],
    toolCallId: null,
    sequence,
    metadata: null,
  };
}

function prompt(text: string) {
  return {
    id: `local:${text}`,
    role: "user" as const,
    parts: [{ type: "text" as const, text }],
  };
}

describe("lastReplySequence", () => {
  const OWN = new Set(["x"]);

  const row = (role: string, sequence: number): PersistedRow => ({
    role,
    sequence,
    content: "x",
  });

  it("names the last reply after the turn's latest user row", () => {
    expect(
      lastReplySequence(
        [
          row("user", 10),
          row("assistant", 11),
          row("tool", 12),
          row("assistant", 13),
          row("tool", 14),
        ],
        OWN,
      ),
    ).toBe(13);
  });

  it("names nothing while the latest user row has no saved reply yet", () => {
    expect(
      lastReplySequence(
        [row("user", 10), row("assistant", 11), row("user", 12)],
        OWN,
      ),
    ).toBeNull();
  });

  it("names nothing when the window holds another turn's question", () => {
    expect(
      lastReplySequence(
        [
          row("assistant", 11),
          { role: "user", sequence: 12, content: "From another tab" },
          row("assistant", 13),
        ],
        OWN,
      ),
    ).toBeNull();
  });
});
