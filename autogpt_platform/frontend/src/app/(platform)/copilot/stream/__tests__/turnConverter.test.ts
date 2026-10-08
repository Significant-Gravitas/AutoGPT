import { describe, expect, it } from "vitest";

import { applyEntry, foldEntries, type WireChunk } from "../turnConverter";
import {
  COPILOT_ERROR_PREFIX,
  COPILOT_RETRYABLE_ERROR_PREFIX,
  emptyTurnLog,
  seedTurnLog,
  type TurnLog,
} from "../turnLog";

const TURN = "turn-1";

function fold(chunks: WireChunk[], log: TurnLog = emptyTurnLog()) {
  const offset = log.cursor ? Number(log.cursor.split("-")[0]) : 0;
  return foldEntries(
    log,
    chunks.map((chunk, i) => ({
      turn: TURN,
      entryId: `${offset + i + 1}-0`,
      chunk,
    })),
  );
}

function rowsOf(log: TurnLog) {
  return log.rows.map((r) => ({
    role: r.role,
    content: r.content,
    calls: r.toolCalls.map((c) => c.id),
    ...(r.toolCallId ? { toolCallId: r.toolCallId } : {}),
  }));
}

function reasons(log: TurnLog) {
  return log.protocolErrors.map((e) => `${e.chunkType}: ${e.reason}`);
}

const text = (id: string, delta: string): WireChunk[] => [
  { type: "text-start", id },
  { type: "text-delta", id, delta },
  { type: "text-end", id },
];
const call = (id: string, input: unknown = {}): WireChunk[] => [
  { type: "tool-input-start", toolCallId: id, toolName: "bash_exec" },
  {
    type: "tool-input-available",
    toolCallId: id,
    toolName: "bash_exec",
    input,
  },
];
const output = (id: string, value: unknown = "ok"): WireChunk => ({
  type: "tool-output-available",
  toolCallId: id,
  output: value,
});

describe("the converter table", () => {
  it("start opens the turn; a second start changes nothing", () => {
    const log = fold([{ type: "start", messageId: "m1" }]);
    expect(log.status).toBe("running");
    expect(log.messageId).toBe("m1");
    expect(fold([{ type: "start", messageId: "m1" }], log)).toMatchObject({
      messageId: "m1",
      rows: [],
    });
  });

  it("a start for a second message in the turn opens a new assistant row", () => {
    const log = fold([
      { type: "start", messageId: "m1" },
      ...text("a", "First reply."),
      { type: "start", messageId: "m2" },
      ...text("b", "Second reply."),
    ]);
    expect(rowsOf(log)).toEqual([
      { role: "assistant", content: "First reply.", calls: [] },
      { role: "assistant", content: "Second reply.", calls: [] },
    ]);
  });

  it("start-step opens a step and finish-step closes it", () => {
    const opened = fold([{ type: "start-step" }]);
    expect(opened.openStep).not.toBeNull();
    expect(fold([{ type: "finish-step" }], opened).openStep).toBeNull();
  });

  it("finish-step with a block still open closes it and counts a protocol error", () => {
    const log = fold([
      { type: "start-step" },
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "hi" },
      { type: "finish-step" },
    ]);
    expect(log.blocks.a.open).toBe(false);
    expect(reasons(log)).toEqual(["finish-step: block open"]);
  });

  it("a known text-start continues the block instead of opening a second one", () => {
    const log = fold([
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "one " },
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "two" },
    ]);
    expect(rowsOf(log)).toEqual([
      { role: "assistant", content: "one two", calls: [] },
    ]);
    expect(log.protocolErrors).toEqual([]);
  });

  it("text opens its row on the first delta, so an empty block persists nothing", () => {
    expect(
      fold([
        { type: "text-start", id: "a" },
        { type: "text-end", id: "a" },
      ]).rows,
    ).toEqual([]);
  });

  it("reasoning-start appends its own row at once", () => {
    const log = fold([
      { type: "reasoning-start", id: "r" },
      { type: "reasoning-delta", id: "r", delta: "think" },
      { type: "reasoning-end", id: "r" },
    ]);
    expect(rowsOf(log)).toEqual([
      { role: "reasoning", content: "think", calls: [] },
    ]);
  });

  it.each([
    [
      "text-delta",
      "closed block",
      [...text("a", "x"), { type: "text-delta", id: "a", delta: "y" }],
    ],
    [
      "text-delta",
      "unknown block",
      [{ type: "text-delta", id: "nope", delta: "y" }],
    ],
    [
      "reasoning-delta",
      "closed block",
      [
        { type: "reasoning-start", id: "r" },
        { type: "reasoning-end", id: "r" },
        { type: "reasoning-delta", id: "r", delta: "y" },
      ],
    ],
    [
      "reasoning-delta",
      "unknown block",
      [{ type: "reasoning-delta", id: "nope", delta: "y" }],
    ],
    [
      "text-delta",
      "unknown block",
      [
        { type: "reasoning-start", id: "r" },
        { type: "text-delta", id: "r", delta: "y" },
      ],
    ],
    ["text-end", "unknown block", [{ type: "text-end", id: "nope" }]],
    ["reasoning-end", "unknown block", [{ type: "reasoning-end", id: "nope" }]],
  ] as [string, string, WireChunk[]][])(
    "%s on a %s is a protocol error that changes no row",
    (type, reason, chunks) => {
      const before = fold(chunks.slice(0, -1));
      const after = fold(chunks.slice(-1), before);
      expect(reasons(after)).toEqual([`${type}: ${reason}`]);
      expect(after.rows).toBe(before.rows);
    },
  );

  it("an end on a closed block is a no-op", () => {
    const log = fold([...text("a", "x"), { type: "text-end", id: "a" }]);
    expect(log.protocolErrors).toEqual([]);
  });

  it("tool-input-available opens the call on the current assistant row", () => {
    const log = fold([...text("a", "Let me look."), ...call("c1", { q: 1 })]);
    expect(rowsOf(log)).toEqual([
      { role: "assistant", content: "Let me look.", calls: ["c1"] },
    ]);
    expect(log.tools.c1.phase).toBe("input-available");
  });

  it("a repeated input-available with the same input is a no-op", () => {
    const once = fold(call("c1", { q: 1 }));
    const twice = fold(call("c1", { q: 1 }), once);
    expect(twice.rows).toBe(once.rows);
    expect(twice.protocolErrors).toEqual([]);
  });

  it("a second input-available with a different input is a protocol error", () => {
    const log = fold([
      ...call("c1", { q: 1 }),
      ...call("c1", { q: 2 }).slice(1),
    ]);
    expect(reasons(log)).toEqual(["tool-input-available: input changed"]);
    expect(log.rows[0].toolCalls[0].input).toEqual({ q: 1 });
  });

  it("tool-output-available sets the result row once; a repeat is a no-op", () => {
    const once = fold([...call("c1"), output("c1", "first")]);
    const twice = fold([output("c1", "second")], once);
    expect(rowsOf(twice)).toEqual([
      { role: "assistant", content: "", calls: ["c1"] },
      { role: "tool", content: "first", calls: [], toolCallId: "c1" },
    ]);
    expect(twice.rows).toBe(once.rows);
  });

  it("an output for an unknown call is a protocol error, and still the row the backend persists", () => {
    const log = fold([output("ghost", { todos: [] })]);
    expect(reasons(log)).toEqual(["tool-output-available: unknown tool call"]);
    expect(rowsOf(log)).toEqual([
      { role: "tool", content: '{"todos":[]}', calls: [], toolCallId: "ghost" },
    ]);
  });

  it.each([
    [
      "[code:transient_api_error] Try again",
      COPILOT_RETRYABLE_ERROR_PREFIX,
      "Try again",
    ],
    ["[code:baseline_error] Boom", COPILOT_RETRYABLE_ERROR_PREFIX, "Boom"],
    ["[code:sdk_stream_error] Boom", COPILOT_ERROR_PREFIX, "Boom"],
    ["No code at all", COPILOT_ERROR_PREFIX, "No code at all"],
  ])(
    "error %j appends a marker row decided by its code",
    (errorText, prefix, shown) => {
      const log = fold([...text("a", "partial"), { type: "error", errorText }]);
      expect(log.rows[1].content).toBe(`${prefix} ${shown}`);
      expect(log.status).toBe("failed");
    },
  );

  it("a provider failure ahead of the error decides the marker and rides on it", () => {
    const failure = {
      kind: "usage_limit",
      message: "Limit reached.",
      retryable: false,
    };
    const log = fold([
      { type: "data-provider-failure", data: failure },
      { type: "error", errorText: "[code:usage_limit] raw" },
    ]);
    expect(log.rows[0]).toMatchObject({
      content: `${COPILOT_ERROR_PREFIX} Limit reached.`,
      metadata: { provider_failure: failure },
    });
    expect(log.overlay.map((p) => p.type)).toEqual(["data-provider-failure"]);
  });

  it("a second error does not stack a second marker", () => {
    const log = fold([
      { type: "error", errorText: "one" },
      { type: "error", errorText: "two" },
    ]);
    expect(log.rows).toHaveLength(1);
  });

  it("finish closes the turn and nothing else", () => {
    const log = fold([...call("c1"), { type: "finish" }]);
    expect(log.status).toBe("finished");
    expect(log.tools.c1.phase).toBe("input-available");
    expect(
      fold([{ type: "error", errorText: "x" }, { type: "finish" }]).status,
    ).toBe("failed");
  });

  it("data-tool-display names the call, before or after it arrives", () => {
    const display = (name: string): WireChunk => ({
      type: "data-tool-display",
      id: "c1",
      data: { toolCallId: "c1", displayName: name },
    });
    expect(
      fold([display("Early"), ...call("c1")]).rows[0].toolCalls[0].displayName,
    ).toBe("Early");
    expect(
      fold([...call("c1"), display("Late")]).rows[0].toolCalls[0].displayName,
    ).toBe("Late");
  });

  it("data-pending-drained appends user rows keyed by pending id, once each", () => {
    const drained: WireChunk = {
      type: "data-pending-drained",
      data: { drainedCount: 1, messages: [{ id: "p1", content: "and also" }] },
    };
    const log = fold([...text("a", "hi"), drained, drained]);
    expect(rowsOf(log).map((r) => r.role)).toEqual(["assistant", "user"]);
    expect(log.rows[1].key).toBe("user:p1");
  });

  it("other data parts go to the overlay, keyed by entry id", () => {
    const log = fold([
      { type: "data-status", data: { message: "Working…" } },
      { type: "data-compaction", data: { phase: "summarizing" } },
    ]);
    expect(log.overlay).toEqual([
      {
        entryId: "1-0",
        type: "data-status",
        data: { message: "Working…" },
        anchor: 0,
      },
      {
        entryId: "2-0",
        type: "data-compaction",
        data: { phase: "summarizing" },
        anchor: 0,
      },
    ]);
    expect(log.rows).toEqual([]);
  });

  it("data-checkpoint assigns sequences to the rows it covers and keeps the digest input", () => {
    const log = fold([
      ...text("a", "hi"),
      ...call("c1"),
      output("c1"),
      {
        type: "data-checkpoint",
        data: { rows: 2, sequence: 7, digest: "d" },
        transient: true,
      },
    ]);
    expect(log.rows.map((r) => r.sequence)).toEqual([7, 8]);
    expect(log.checkpoints).toEqual([
      {
        entryId: "7-0",
        rows: 2,
        sequence: 7,
        digest: "d",
        canonical: '[["assistant","hi",[["c1","bash_exec"]]],["tool","c1",[]]]',
      },
    ]);
  });

  it("an unknown chunk type is a protocol error", () => {
    expect(reasons(fold([{ type: "tool-input-delta" }]))).toEqual([
      "tool-input-delta: unknown chunk type",
    ]);
  });
});

describe("the cursor guard", () => {
  it("drops an entry at or before the cursor, so an overlapping replay applies nothing twice", () => {
    const entries = [...text("a", "hello"), ...call("c1"), output("c1")].map(
      (chunk, i) => ({ turn: TURN, entryId: `${i + 1}-0`, chunk }),
    );
    const once = foldEntries(emptyTurnLog(), entries);
    const replayed = foldEntries(once, entries);
    expect(replayed).toBe(once);
    expect(rowsOf(replayed)).toEqual([
      { role: "assistant", content: "hello", calls: ["c1"] },
      { role: "tool", content: "ok", calls: [], toolCallId: "c1" },
    ]);
  });

  it("orders ids numerically on both halves", () => {
    const log = foldEntries(emptyTurnLog(), [
      { turn: TURN, entryId: "10-2", chunk: { type: "text-start", id: "a" } },
      {
        turn: TURN,
        entryId: "10-10",
        chunk: { type: "text-delta", id: "a", delta: "x" },
      },
      {
        turn: TURN,
        entryId: "9-99",
        chunk: { type: "text-delta", id: "a", delta: "dup" },
      },
    ]);
    expect(log.rows[0].content).toBe("x");
  });

  it("an entry from another turn is a protocol error and moves nothing", () => {
    const log = fold(text("a", "x"));
    const other = applyEntry(log, {
      turn: "turn-2",
      entryId: "99-0",
      chunk: { type: "text-delta", id: "a", delta: "y" },
    });
    expect(reasons(other)).toEqual(["text-delta: another turn"]);
    expect(other.cursor).toBe(log.cursor);
  });
});

describe("the row rules", () => {
  it("after tool results, the next text or call opens a new assistant row", () => {
    const log = fold([
      ...text("a", "First."),
      ...call("c1"),
      output("c1"),
      ...call("c2"),
      output("c2"),
      ...text("b", "Then."),
    ]);
    expect(rowsOf(log)).toEqual([
      { role: "assistant", content: "First.", calls: ["c1"] },
      { role: "tool", content: "ok", calls: [], toolCallId: "c1" },
      { role: "assistant", content: "", calls: ["c2"] },
      { role: "tool", content: "ok", calls: [], toolCallId: "c2" },
      { role: "assistant", content: "Then.", calls: [] },
    ]);
  });

  it("text after a reasoning block with no tool result between extends the earlier assistant row", () => {
    const log = fold([
      ...text("a", "One. "),
      { type: "reasoning-start", id: "r" },
      { type: "reasoning-delta", id: "r", delta: "hmm" },
      { type: "reasoning-end", id: "r" },
      ...text("b", "Two."),
    ]);
    expect(rowsOf(log)).toEqual([
      { role: "assistant", content: "One. Two.", calls: [] },
      { role: "reasoning", content: "hmm", calls: [] },
    ]);
  });

  it("seeding from persisted rows at a checkpoint continues where the stream does", () => {
    const seeded = seedTurnLog({
      turnId: TURN,
      rows: [
        { role: "user", content: "prompt", sequence: 0 },
        {
          role: "assistant",
          content: "Looking.",
          sequence: 1,
          tool_calls: [
            { id: "c1", function: { name: "bash_exec", arguments: "{}" } },
          ],
        },
        { role: "tool", content: "ok", tool_call_id: "c1", sequence: 2 },
      ],
      checkpoint: { entry_id: "5-0", rows: 2, sequence: 1 },
    });
    expect(seeded.hasToolResults).toBe(true);
    const log = fold(text("b", "Done."), seeded);
    expect(rowsOf(log)).toEqual([
      { role: "assistant", content: "Looking.", calls: ["c1"] },
      { role: "tool", content: "ok", calls: [], toolCallId: "c1" },
      { role: "assistant", content: "Done.", calls: [] },
    ]);
    expect(log.rows.map((r) => r.sequence)).toEqual([1, 2, null]);
  });
});
