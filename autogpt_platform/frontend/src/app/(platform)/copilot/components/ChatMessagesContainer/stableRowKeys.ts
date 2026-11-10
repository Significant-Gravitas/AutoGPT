import type { UIDataTypes, UIMessage, UITools } from "ai";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

/** Row keys that differ from the row's message id, by message id. */
export type RowKeys = ReadonlyMap<string, string>;

/**
 * Carries each row's React key across an update that replaces message ids.
 *
 * When a stream ends, hydration swaps the turn's client ids (the prompt's
 * local id, the AI SDK's message id) for the stored rows' `<session>-seq-N`
 * ids. The ids must change, because ratings, sequences and the turn stats are
 * read from them, but a row keyed by its id remounts: the entrance animation
 * replays, the markdown is parsed again and an expanded chain collapses.
 *
 * So a row that appears in the same update another one vanishes in takes
 * that row's key: in order, each new row pairs with the next vanished row of
 * its role. A row that stays keeps its key, and any other row is keyed by its
 * id. A key two rows would share stays only with the first of them.
 */
export function carryRowKeys(
  prevRows: readonly ChatMessage[],
  prevKeys: RowKeys,
  nextRows: readonly ChatMessage[],
): RowKeys {
  const prevIDs = new Set(prevRows.map((row) => row.id));
  const nextIDs = new Set(nextRows.map((row) => row.id));
  const vanished = prevRows.filter((row) => !nextIDs.has(row.id));
  const keys = new Map<string, string>();
  let from = 0;
  for (const row of nextRows) {
    if (prevIDs.has(row.id)) {
      const kept = prevKeys.get(row.id);
      if (kept) keys.set(row.id, kept);
      continue;
    }
    const at = vanished.findIndex((v, i) => i >= from && v.role === row.role);
    if (at === -1) continue;
    from = at + 1;
    const key = prevKeys.get(vanished[at].id) ?? vanished[at].id;
    if (key !== row.id) keys.set(row.id, key);
  }
  return withoutSharedKeys(nextRows, keys);
}

export function rowKeyOf(keys: RowKeys, id: string): string {
  return keys.get(id) ?? id;
}

function withoutSharedKeys(rows: readonly ChatMessage[], keys: RowKeys) {
  const used = new Set<string>();
  const unique = new Map<string, string>();
  for (const row of rows) {
    const key = rowKeyOf(keys, row.id);
    if (used.has(key)) continue;
    used.add(key);
    if (key !== row.id) unique.set(row.id, key);
  }
  return unique;
}
