import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { v5 as uuidv5 } from "uuid";
import { describe, expect, it } from "vitest";

import { emptyTurnLog, type LogRow, type PersistedRow } from "../turnLog";
import {
  gapBeforePrompt,
  gapBeforeTurn,
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

describe("gapBeforePrompt", () => {
  // Another tab's turn (2, 3) landed before this tab's prompt (4).
  const ROWS: PersistedRow[] = [
    { role: "user", content: "lima", sequence: 0 },
    { role: "assistant", content: "Lima.", sequence: 1 },
    { role: "user", content: "mike", sequence: 2 },
    { role: "assistant", content: "Mike.", sequence: 3 },
    { role: "user", content: "november", sequence: 4 },
    { role: "assistant", content: "November.", sequence: 5 },
  ];
  const sent = (text: string, sequence: number | null) => ({
    ...userSegment(prompt(text), "sent"),
    sequence,
  });

  it("takes the rows between where the tail starts and a prompt that opens it", () => {
    // Sent from a view that ended at 1, so the tail claimed 2 onwards.
    const gap = gapBeforePrompt([sent("november", 4)], 0, 2, ROWS, -Infinity);

    expect(gap?.key).toBe("rows:2");
    expect(gap?.rows.map((r) => r.content)).toEqual(["mike", "Mike."]);
  });

  it("takes the rows between the tail's last turn and the next prompt", () => {
    const answered = newTurnSegment("turn-1", {
      ...emptyTurnLog(),
      rows: [row("block-1", "Lima.", 1)],
    });
    const segments: Segment[] = [
      sent("lima", 0),
      answered,
      sent("november", 4),
    ];

    const gap = gapBeforePrompt(segments, 2, 0, ROWS, -Infinity);

    expect(gap?.rows.map((r) => r.sequence)).toEqual([2, 3]);
  });

  it("takes nothing when the prompt follows the tail at once", () => {
    expect(gapBeforePrompt([sent("mike", 2)], 0, 2, ROWS, -Infinity)).toBe(
      null,
    );
  });

  it("takes nothing while the segment before ends at no known row", () => {
    // A turn not reconciled yet: its rows have no sequences.
    const streaming = newTurnSegment("turn-1", {
      ...emptyTurnLog(),
      rows: [row("block-1", "Lima.", null)],
    });
    const segments: Segment[] = [
      sent("lima", 0),
      streaming,
      sent("november", 4),
    ];

    expect(gapBeforePrompt(segments, 2, 0, ROWS, -Infinity)).toBe(null);
  });

  it("takes nothing for a prompt with no row yet", () => {
    expect(
      gapBeforePrompt([sent("november", null)], 0, 2, ROWS, -Infinity),
    ).toBe(null);
  });

  it("takes nothing from a session window that starts inside the gap", () => {
    const windowed = ROWS.slice(3);

    expect(gapBeforePrompt([sent("november", 4)], 0, 2, windowed, 3)).toBe(
      null,
    );
    expect(
      gapBeforePrompt([sent("november", 4)], 0, 2, ROWS.slice(2), 2)?.rows,
    ).toHaveLength(2);
  });
});

describe("gapBeforeTurn", () => {
  // The tail ends at 1. Another tab's turn (2, 3) came before the running
  // turn's prompt (4); that turn's rows start at 5.
  const ROWS: PersistedRow[] = [
    { role: "user", content: "lima", sequence: 0 },
    { role: "assistant", content: "Lima.", sequence: 1 },
    { role: "user", content: "mike", sequence: 2 },
    { role: "assistant", content: "Mike.", sequence: 3 },
    { role: "user", content: "november", sequence: 4 },
    { role: "assistant", content: "November.", sequence: 5 },
  ];
  const tail = (sequence: number | null = 1): Segment[] => [
    { ...userSegment(prompt("lima"), "sent"), sequence: 0 },
    newTurnSegment("turn-1", {
      ...emptyTurnLog(),
      rows: [row("block-1", "Lima.", sequence)],
    }),
  ];
  const sequences = (rows: readonly PersistedRow[] | undefined) =>
    rows?.map((r) => r.sequence);

  it("takes every row between the tail and the turn from a window that reaches the tail's end", () => {
    expect(sequences(gapBeforeTurn(tail(), 5, ROWS.slice(2), 2)?.rows)).toEqual(
      [2, 3, 4],
    );
  });

  it("takes only the turn's prompt from a window that starts past the tail's end", () => {
    expect(sequences(gapBeforeTurn(tail(), 5, ROWS.slice(3), 3)?.rows)).toEqual(
      [4],
    );
  });

  it("takes the follow-ups queued before the turn's prompt as well", () => {
    const queued: PersistedRow[] = [
      { role: "assistant", content: "Mike.", sequence: 3 },
      { role: "user", content: "oscar", sequence: 4 },
      { role: "user", content: "papa", sequence: 5 },
      { role: "assistant", content: "Papa.", sequence: 6 },
    ];

    expect(sequences(gapBeforeTurn(tail(), 6, queued, 3)?.rows)).toEqual([
      4, 5,
    ]);
  });

  it("takes nothing from a window that starts past the turn's prompt", () => {
    expect(gapBeforeTurn(tail(), 5, ROWS.slice(5), 5)).toBe(null);
  });

  it("takes nothing while the tail ends at no known row", () => {
    expect(gapBeforeTurn(tail(null), 5, ROWS.slice(3), 3)).toBe(null);
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
