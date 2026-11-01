import { isUnsafeIntegerText } from "./integer-input";

/**
 * True when `text` is JSON for `value`, however it is spaced or indented.
 * A field that edits JSON as text keeps the user's own text while it still
 * says the same thing as the stored value. Replacing it with a reformatted
 * copy rewrites the field under the user, which throws the caret to the end
 * and cancels an IME composition.
 *
 * Key order counts, which is fine for a value parsed from the user's own
 * text; a value with its keys in another order only costs one reformat.
 */
export function isJsonTextFor(text: string, value: unknown): boolean {
  if (text.trim() === "") return value === undefined || value === null;
  if (hasNonFiniteNumber(text)) return false;
  try {
    return JSON.stringify(parseJsonExact(text)) === JSON.stringify(value);
  } catch {
    return false;
  }
}

const JSON_TOKEN = /"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
// The smallest unsafe integer, 2^53, has 16 digits.
const LONG_DIGIT_RUN = /\d{16}/;

/**
 * JSON.parse, except integers beyond Number.MAX_SAFE_INTEGER come back as
 * their digit string instead of a rounded number. The backend converts the
 * string to its int when the field is typed as one.
 */
export function parseJsonExact(text: string): unknown {
  if (!LONG_DIGIT_RUN.test(text)) return JSON.parse(text);
  return JSON.parse(
    text.replace(JSON_TOKEN, (token) =>
      isUnsafeIntegerText(token) ? `"${token}"` : token,
    ),
  );
}

// A number too large to hold at all (1e400) parses to Infinity, which is
// saved as null: something other than what is on screen.
function hasNonFiniteNumber(text: string): boolean {
  return (text.match(JSON_TOKEN) ?? []).some(
    (token) => !token.startsWith('"') && !Number.isFinite(Number(token)),
  );
}
