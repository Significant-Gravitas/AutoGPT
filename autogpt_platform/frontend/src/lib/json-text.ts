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

const JSON_TOKEN =
  /"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
// The smallest unsafe integer, 2^53, has 16 digits.
const LONG_DIGIT_RUN = /\d{16}/;

// Marks a big integer while JSON.parse runs, so it can be told apart from a
// string that holds the same digits.
const MARK = "\u0000";

// `${key}${MARK}${digits}` for each big integer read from a JSON number.
// Kept in the browser only: on the server it would be shared across users.
const integerLiterals = new Set<string>();

/**
 * JSON.parse, except integers beyond Number.MAX_SAFE_INTEGER come back as
 * their digit string instead of a rounded number. The backend converts the
 * string to its int when the field is typed as one; for any other field,
 * restoreIntegerLiterals sends it back as the number it was read as.
 */
export function parseJsonExact(text: string): unknown {
  if (!LONG_DIGIT_RUN.test(text)) return JSON.parse(text);
  const marked = text.replace(JSON_TOKEN, (token) =>
    isUnsafeIntegerText(token) ? `"\\u0000${token}"` : token,
  );
  return JSON.parse(marked, (key, value) => {
    if (typeof value !== "string" || !value.startsWith(MARK)) return value;
    const digits = value.slice(1);
    if (!isUnsafeIntegerText(digits)) return value;
    if (typeof window !== "undefined") {
      integerLiterals.add(key + MARK + digits);
    }
    return digits;
  });
}

/**
 * Writes each digit string that parseJsonExact read from a JSON number under
 * the same key back as that number, so an untouched value keeps its type.
 * Digits stored or typed as a string stay a string unless the same key held
 * them as a number in something loaded before.
 */
export function restoreIntegerLiterals(json: string): string {
  if (integerLiterals.size === 0 || !LONG_DIGIT_RUN.test(json)) return json;
  const marked = JSON.parse(json, (key, value) =>
    typeof value === "string" && isLoadedInteger(key, value)
      ? MARK + value
      : value,
  );
  return JSON.stringify(marked).replace(/"\\u0000(-?\d+)"/g, "$1");
}

// Review approval sends the loaded payload under reviewed_data.
function isLoadedInteger(key: string, digits: string): boolean {
  if (integerLiterals.has(key + MARK + digits)) return true;
  return (
    key === "reviewed_data" && integerLiterals.has("payload" + MARK + digits)
  );
}

// A number too large to hold at all (1e400) parses to Infinity, which is
// saved as null: something other than what is on screen.
function hasNonFiniteNumber(text: string): boolean {
  return (text.match(JSON_TOKEN) ?? []).some(
    (token) => !token.startsWith('"') && !Number.isFinite(Number(token)),
  );
}
