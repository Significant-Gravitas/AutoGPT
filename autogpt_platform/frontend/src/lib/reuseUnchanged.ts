function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * Returns `next`, but keeps every part of `previous` that is deep-equal to the
 * matching part of `next`, so unchanged subtrees keep their references. When
 * the two are deep-equal the result is `previous` itself, which makes
 * `reuseUnchanged(a, b) === a` a cheap equality check.
 *
 * Plain objects and arrays are compared by content and dates by time. Other
 * objects (Map, Set, class instances) are only equal when they are the same
 * reference. Unlike React Query's structural sharing, dates count as equal,
 * since API responses turn every timestamp into a fresh `Date`.
 */
export function reuseUnchanged<T>(previous: unknown, next: T): T {
  if (Object.is(previous, next)) return previous as T;

  if (previous instanceof Date && next instanceof Date) {
    return Object.is(previous.getTime(), next.getTime())
      ? (previous as T)
      : next;
  }

  if (Array.isArray(previous) && Array.isArray(next)) {
    let changed = previous.length !== next.length;
    const merged = next.map((item, index) => {
      const kept = reuseUnchanged(previous[index], item);
      if (!Object.is(kept, previous[index])) changed = true;
      return kept;
    });
    return (changed ? merged : previous) as T;
  }

  if (isPlainObject(previous) && isPlainObject(next)) {
    const keys = Object.keys(next);
    let changed = keys.length !== Object.keys(previous).length;
    const merged: Record<string, unknown> = {};
    for (const key of keys) {
      const kept = reuseUnchanged(previous[key], next[key]);
      if (!Object.is(kept, previous[key]) || !Object.hasOwn(previous, key)) {
        changed = true;
      }
      merged[key] = kept;
    }
    return (changed ? merged : previous) as T;
  }

  return next;
}
