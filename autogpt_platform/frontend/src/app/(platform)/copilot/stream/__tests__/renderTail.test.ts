import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";

import {
  createAssistantRunJoiner,
  createTailRenderer,
  statsOfAssistantRuns,
} from "../renderTail";
import type { RuntimeSnapshot } from "../turnRuntime";
import {
  emptyTurnLog,
  logRowFromPersisted,
  type LogRow,
  type PersistedRow,
} from "../turnLog";
import {
  newTurnSegment,
  reconcileTurn,
  userSegment,
  type Segment,
} from "../turnTail";

type Msg = UIMessage<unknown, UIDataTypes, UITools>;

function message(id: string, role: Msg["role"]): Msg {
  return { id, role, parts: [{ type: "text", text: id }] };
}

describe("statsOfAssistantRuns", () => {
  it("gives a joined bubble its last turn's saved stats, as a reload does", () => {
    const messages = [
      message("u1", "user"),
      message("a1", "assistant"),
      message("a2", "assistant"),
      message("u2", "user"),
      message("a3", "assistant"),
    ];
    const stats = new Map([
      ["a1", { durationMs: 5_000, createdAt: "2026-10-24T10:00:00Z" }],
      ["a2", { durationMs: 30_000, createdAt: "2026-10-24T10:01:00Z" }],
      ["a3", { durationMs: 7_000 }],
    ]);

    const bubbles = createAssistantRunJoiner()(messages);
    const joined = statsOfAssistantRuns(messages, stats);

    expect(bubbles.map((bubble) => bubble.id)).toEqual([
      "u1",
      "a1",
      "u2",
      "a3",
    ]);
    expect(joined.get("a1")).toEqual({
      durationMs: 30_000,
      createdAt: "2026-10-24T10:01:00Z",
      savedMessageId: null,
    });
    // A user row between them keeps two bubbles apart.
    expect(joined.get("a3")).toEqual({ durationMs: 7_000 });
  });

  it("keeps a saved history bubble from rating a later turn that joins it", () => {
    const history = message("sess-1-seq-5", "assistant");
    const messages = [history, message("turn:turn-2", "assistant")];

    const joined = statsOfAssistantRuns(messages, new Map());

    expect(joined.get("sess-1-seq-5")?.savedMessageId).toBeNull();
  });

  it("rates two joined history bubbles by the later one's id", () => {
    const messages = [
      message("sess-1-seq-5", "assistant"),
      message("sess-1-seq-9", "assistant"),
    ];

    const joined = statsOfAssistantRuns(messages, new Map());

    expect(joined.get("sess-1-seq-5")?.savedMessageId).toBe("sess-1-seq-9");
  });
});

const SESSION = "sess-1";

function snapshot(segments: Segment[]): RuntimeSnapshot {
  return {
    segments,
    ownedFrom: 0,
    phase: "idle",
    notice: null,
    error: null,
    stopped: false,
  };
}

function streamedRow(
  key: string,
  role: LogRow["role"],
  content: string,
): LogRow {
  return {
    key,
    role,
    content,
    toolCalls: [],
    toolCallId: null,
    sequence: null,
    metadata: null,
  };
}

function finishedTurn() {
  return newTurnSegment("turn-1", {
    ...emptyTurnLog(),
    turnId: "turn-1",
    status: "finished",
    rows: [
      streamedRow("block-r", "reasoning", "Looking it up"),
      streamedRow("block-1", "assistant", "Here is the summary."),
    ],
  });
}

const PERSISTED: PersistedRow[] = [
  { role: "reasoning", content: "Looking it up", sequence: 4 },
  { role: "assistant", content: "Here is the summary.", sequence: 5 },
];

const PROMPT = {
  ...userSegment(
    { id: "local:hi", role: "user", parts: [{ type: "text", text: "hi" }] },
    "sent",
  ),
  sequence: 3,
};

describe("createTailRenderer saved bubble ids", () => {
  it("names a finished turn's bubble by its last saved row once reconciled", () => {
    const reconciled = reconcileTurn(finishedTurn(), {
      raw: PERSISTED,
      rows: PERSISTED.map(logRowFromPersisted),
    });

    const { messages, stats } = createTailRenderer(SESSION)(
      snapshot([PROMPT, reconciled]),
    );

    const bubble = messages.find((m) => m.role === "assistant");
    // The bubble keeps the turn's key, so React does not remount it.
    expect(bubble?.id).toBe("turn:turn-1");
    expect(stats.get("turn:turn-1")?.savedMessageId).toBe(`${SESSION}-seq-5`);
  });

  it("gives a turn that is not reconciled yet no saved id", () => {
    const { stats } = createTailRenderer(SESSION)(
      snapshot([PROMPT, finishedTurn()]),
    );

    expect(stats.get("turn:turn-1")?.savedMessageId).toBeUndefined();
  });

  it("lets a joined bubble's later saved row win, as a reload keys it", () => {
    const reconciled = reconcileTurn(finishedTurn(), {
      raw: PERSISTED,
      rows: PERSISTED.map(logRowFromPersisted),
    });
    const later: Segment = {
      kind: "rows",
      key: "rows:6",
      rows: [{ role: "assistant", content: "One more thing.", sequence: 6 }],
    };

    const { messages, stats } = createTailRenderer(SESSION)(
      snapshot([PROMPT, reconciled, later]),
    );
    const joined = statsOfAssistantRuns(messages, stats);

    expect(createAssistantRunJoiner()(messages).map((m) => m.id)).toEqual([
      "local:hi",
      "turn:turn-1",
    ]);
    expect(joined.get("turn:turn-1")?.savedMessageId).toBe(`${SESSION}-seq-6`);
  });

  it("rates a joined bubble by nothing until its later turn is saved", () => {
    const reconciled = reconcileTurn(finishedTurn(), {
      raw: PERSISTED,
      rows: PERSISTED.map(logRowFromPersisted),
    });
    const continuation = newTurnSegment("turn-2", {
      ...emptyTurnLog(),
      turnId: "turn-2",
      status: "finished",
      rows: [streamedRow("block-2", "assistant", "Picked it back up.")],
    });
    const render = createTailRenderer(SESSION);

    const live = render(snapshot([PROMPT, reconciled, continuation]));
    expect(
      statsOfAssistantRuns(live.messages, live.stats).get("turn:turn-1")
        ?.savedMessageId,
    ).toBeNull();

    const saved = reconcileTurn(continuation, {
      raw: [{ role: "assistant", content: "Picked it back up.", sequence: 6 }],
      rows: [
        logRowFromPersisted({
          role: "assistant",
          content: "Picked it back up.",
          sequence: 6,
        }),
      ],
    });
    const done = render(snapshot([PROMPT, reconciled, saved]));
    expect(
      statsOfAssistantRuns(done.messages, done.stats).get("turn:turn-1")
        ?.savedMessageId,
    ).toBe(`${SESSION}-seq-6`);
  });

  it("names a turn too long to reconcile by its last saved reply row", () => {
    const longTurn = { ...finishedTurn(), lastReplySequence: 87 };

    const { stats } = createTailRenderer(SESSION)(snapshot([PROMPT, longTurn]));

    expect(stats.get("turn:turn-1")?.savedMessageId).toBe(`${SESSION}-seq-87`);
  });
});
