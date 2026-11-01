const INTEGER_TEXT = /^-?\d+$/;

/** True for integer text a JS number can't hold exactly. */
export function isUnsafeIntegerText(text: string): boolean {
  return INTEGER_TEXT.test(text) && !Number.isSafeInteger(Number(text));
}

/**
 * Parses integer field text. Integers beyond Number.MAX_SAFE_INTEGER stay as
 * their digit string so no digits are lost; the backend converts it to int.
 */
export function parseIntegerInput(text: string): number | string | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  if (isUnsafeIntegerText(trimmed)) return trimmed;
  const n = Number(trimmed);
  return Number.isFinite(n) ? Math.trunc(n) : undefined;
}
