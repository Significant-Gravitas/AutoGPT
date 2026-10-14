import { describe, expect, it } from "vitest";

import { foldEntries, type WireChunk } from "../../stream/turnConverter";
import { emptyTurnLog, type TurnLog } from "../../stream/turnLog";
import {
  entriesOf,
  loadRecordedTurn,
  recordedTurnNames,
} from "../../stream/__tests__/recordedTurns";
import {
  convertChatSessionMessagesToUiMessages,
  createTurnLogRenderer,
} from "../convertChatSessionToUiMessages";

function fold(chunks: WireChunk[], log: TurnLog = emptyTurnLog()) {
  const offset = log.cursor ? Number(log.cursor.split("-")[0]) : 0;
  return foldEntries(
    log,
    chunks.map((chunk, i) => ({
      turn: "t",
      entryId: `${offset + i + 1}-0`,
      chunk,
    })),
  );
}

const render = (log: TurnLog) => createTurnLogRenderer((id) => id)(log);

// Follow-ups are drained at tool boundaries, so a drain follows a result.
const toolRound: WireChunk[] = [
  {
    type: "tool-input-available",
    toolCallId: "c1",
    toolName: "bash_exec",
    input: {},
  },
  { type: "tool-output-available", toolCallId: "c1", output: "ok" },
];

describe("rendering a turn log", () => {
  it("marks the block still writing as streaming, and done once it ends", () => {
    const open = fold([
      { type: "reasoning-start", id: "r" },
      { type: "reasoning-delta", id: "r", delta: "hmm" },
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "Hel" },
    ]);
    expect(render(open)[0].parts).toMatchObject([
      { type: "reasoning", text: "hmm", state: "streaming" },
      { type: "text", text: "Hel", state: "streaming" },
    ]);
    const closed = fold(
      [
        { type: "reasoning-end", id: "r" },
        { type: "text-end", id: "a" },
      ],
      open,
    );
    expect(render(closed)[0].parts).toMatchObject([
      { state: "done" },
      { state: "done" },
    ]);
  });

  it("shows a pending call as running, then as interrupted once the turn is over", () => {
    const pending = fold([
      {
        type: "tool-input-available",
        toolCallId: "c1",
        toolName: "bash_exec",
        input: { command: "ls" },
      },
    ]);
    expect(render(pending)[0].parts).toEqual([
      expect.objectContaining({
        type: "tool-bash_exec",
        state: "input-available",
        input: { command: "ls" },
      }),
    ]);
    const finished = fold([{ type: "finish" }], pending);
    expect(render(finished)[0].parts).toEqual([
      expect.objectContaining({
        state: "output-error",
        errorText: "Interrupted",
      }),
    ]);
    const answered = fold(
      [
        {
          type: "tool-output-available",
          toolCallId: "c1",
          output: { files: 2 },
        },
      ],
      pending,
    );
    expect(render(answered)[0].parts).toEqual([
      expect.objectContaining({
        state: "output-available",
        output: { files: 2 },
      }),
    ]);
  });

  it("puts a status that arrives before any content in the turn's own bubble, under one key throughout", () => {
    const status = fold([
      { type: "data-status", data: { message: "Message received…" } },
      { type: "start-step" },
    ]);
    const [early] = render(status);
    expect(early.id).toBe("turn:t");
    expect(early.parts).toEqual([
      {
        type: "data-status",
        id: "1-0",
        data: { message: "Message received…" },
      },
      { type: "step-start" },
    ]);
    const withText = fold(
      [
        { type: "text-start", id: "a" },
        { type: "text-delta", id: "a", delta: "Hi" },
      ],
      status,
    );
    const [later] = render(withText);
    expect(later.id).toBe("turn:t");
    expect(later.parts.map((p) => p.type)).toEqual(["data-status", "text"]);
  });

  it("gives a drained follow-up its own bubble and keys the answer after it by that follow-up", () => {
    const log = fold([
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "One." },
      { type: "text-end", id: "a" },
      ...toolRound,
      {
        type: "data-pending-drained",
        data: { messages: [{ id: "p1", content: "and two?" }] },
      },
      { type: "text-start", id: "b" },
      { type: "text-delta", id: "b", delta: "Two." },
    ]);
    expect(render(log).map((m) => [m.id, m.role])).toEqual([
      ["turn:t", "assistant"],
      ["user:p1", "user"],
      ["turn:t:after:user:p1", "assistant"],
    ]);
  });

  it("returns an unchanged bubble as the same object, so React skips it", () => {
    const renderer = createTurnLogRenderer((id) => id);
    const before = fold([
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "One." },
      { type: "text-end", id: "a" },
      ...toolRound,
      {
        type: "data-pending-drained",
        data: { messages: [{ id: "p1", content: "more" }] },
      },
      { type: "text-start", id: "b" },
      { type: "text-delta", id: "b", delta: "Tw" },
    ]);
    const first = renderer(before);
    const second = renderer(
      fold([{ type: "text-delta", id: "b", delta: "o." }], before),
    );
    expect(second[0]).toBe(first[0]);
    expect(second[1]).toBe(first[1]);
    expect(second[2]).not.toBe(first[2]);
    expect(second[2].parts).toMatchObject([{ text: "Two." }]);
  });
});

// The renderer shares its row rules with hydration, so the finished turn a
// user watched stream is the turn a reload of its rows shows.
describe.each(recordedTurnNames())("live equals reload: %s", (name) => {
  it("renders the folded turn as hydration renders its persisted rows", async () => {
    const turn = loadRecordedTurn(name);
    const log = foldEntries(emptyTurnLog(), await entriesOf(turn));
    const first = log.checkpoints[0].sequence;
    const reloaded = convertChatSessionMessagesToUiMessages(
      "s",
      turn.rows.filter((row) => (row.sequence ?? -1) >= first),
      { isComplete: true },
    ).messages;
    const live = render(log);
    expect(
      live.map((m) => m.parts.filter((p) => !p.type.startsWith("data-"))),
    ).toEqual(reloaded.map((m) => m.parts));
  });
});
