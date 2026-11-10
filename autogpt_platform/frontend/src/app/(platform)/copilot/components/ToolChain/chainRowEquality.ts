import type { ChainRow } from "./helpers";

function isShallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  const aKeys = Object.keys(a);
  if (aKeys.length !== Object.keys(b).length) return false;
  return aKeys.every((key) =>
    Object.is(
      (a as Record<string, unknown>)[key],
      (b as Record<string, unknown>)[key],
    ),
  );
}

/** Whether two builds of a row would draw the same thing. The chain rebuilds
 *  every row object on each streamed flush, but a row's input and output keep
 *  their identity across flushes (the SDK copies parts, not their values),
 *  so comparing fields by identity finds the rows that really changed. The
 *  held-call info is rebuilt with the row, so it compares one level down. */
export function isSameChainRow(a: ChainRow, b: ChainRow): boolean {
  if (a === b) return true;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) {
    const field = key as keyof ChainRow;
    const same =
      field === "held"
        ? isShallowEqual(a.held, b.held)
        : Object.is(a[field], b[field]);
    if (!same) return false;
  }
  return true;
}

/** Props comparator for a chain row view: `row` by `isSameChainRow`, every
 *  other prop by identity. */
export function areChainRowPropsEqual<P extends { row: ChainRow }>(
  prev: P,
  next: P,
): boolean {
  const keys = new Set([...Object.keys(prev), ...Object.keys(next)]);
  for (const key of keys) {
    if (key === "row") continue;
    if (!Object.is(prev[key as keyof P], next[key as keyof P])) return false;
  }
  return isSameChainRow(prev.row, next.row);
}
