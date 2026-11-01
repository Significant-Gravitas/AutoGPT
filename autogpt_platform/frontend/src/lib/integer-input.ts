const INTEGER_TEXT = /^-?\d+$/;

/**
 * Parses integer field text. Integers beyond Number.MAX_SAFE_INTEGER stay as
 * their digit string so no digits are lost; the backend converts it to int.
 */
export function parseIntegerInput(text: string): number | string | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  const n = Number(trimmed);
  if (INTEGER_TEXT.test(trimmed) && !Number.isSafeInteger(n)) return trimmed;
  return Number.isFinite(n) ? Math.trunc(n) : undefined;
}
