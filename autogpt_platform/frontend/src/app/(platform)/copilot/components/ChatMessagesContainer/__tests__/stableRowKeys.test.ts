import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import { carryRowKeys, rowKeyOf, type RowKeys } from "../stableRowKeys";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

function row(id: string, role: Message["role"]): Message {
  return { id, role, parts: [] };
}

function keysOf(rows: Message[], keys: RowKeys) {
  return rows.map((r) => rowKeyOf(keys, r.id));
}

describe("carryRowKeys", () => {
  const history = [row("s-seq-1", "user"), row("s-seq-2", "assistant")];

  it("gives the stored rows of a finished turn the keys its live rows had", () => {
    const live = [
      ...history,
      row("local-1", "user"),
      row("uuid-1", "assistant"),
    ];
    const stored = [
      ...history,
      row("s-seq-3", "user"),
      row("s-seq-5", "assistant"),
    ];
    const keys = carryRowKeys(live, new Map(), stored);
    expect(keysOf(stored, keys)).toEqual([
      "s-seq-1",
      "s-seq-2",
      "local-1",
      "uuid-1",
    ]);
  });

  it("keeps a carried key through later updates, and through a second swap", () => {
    const live = [row("local-1", "user"), row("uuid-1", "assistant")];
    const stored = [row("s-seq-1", "user"), row("s-seq-2", "assistant")];
    const first = carryRowKeys(live, new Map(), stored);
    const grown = [...stored, row("local-2", "user")];
    const second = carryRowKeys(stored, first, grown);
    expect(keysOf(grown, second)).toEqual(["local-1", "uuid-1", "local-2"]);
    const merged = [
      row("s-seq-1", "user"),
      row("s-seq-4", "assistant"),
      row("s-seq-5", "user"),
    ];
    const third = carryRowKeys(grown, second, merged);
    expect(keysOf(merged, third)).toEqual(["local-1", "uuid-1", "local-2"]);
  });

  it("pairs rows only with a vanished row of the same role", () => {
    const live = [row("local-1", "user")];
    const stored = [row("s-seq-1", "assistant"), row("s-seq-2", "user")];
    const keys = carryRowKeys(live, new Map(), stored);
    expect(keysOf(stored, keys)).toEqual(["s-seq-1", "local-1"]);
  });

  it("leaves older pages and new rows keyed by their ids", () => {
    const shown = [row("s-seq-5", "user"), row("s-seq-6", "assistant")];
    const paged = [
      row("s-seq-3", "user"),
      row("s-seq-4", "assistant"),
      ...shown,
    ];
    expect(keysOf(paged, carryRowKeys(shown, new Map(), paged))).toEqual([
      "s-seq-3",
      "s-seq-4",
      "s-seq-5",
      "s-seq-6",
    ]);
  });

  it("never gives two rows one key", () => {
    // A row comes back under the id another row took the key of.
    const before = [row("b", "assistant")];
    const prevKeys = new Map([["b", "a"]]);
    const after = [row("a", "assistant"), row("c", "assistant")];
    const keys = keysOf(after, carryRowKeys(before, prevKeys, after));
    expect(new Set(keys).size).toBe(keys.length);
  });
});
