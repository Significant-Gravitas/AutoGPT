/** The arguments of a tool call whose JSON is still streaming, cut down to
 *  the ones that are complete. The model writes keys in order, so every key
 *  but the last one is final; the last may still be growing ("par" on the way
 *  to "paris"). Reading only the finished keys keeps a row's subject from
 *  changing on every delta: it appears once, when its argument is done.
 *  Anything that is not a plain object yet (nothing parsed, a bare string,
 *  an array) has no finished keys. */
export function completeStreamingInput(
  input: unknown,
): Record<string, unknown> | undefined {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return undefined;
  }
  const entries = Object.entries(input as Record<string, unknown>);
  if (entries.length < 2) return undefined;
  return Object.fromEntries(entries.slice(0, -1));
}
