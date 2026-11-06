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
// 16 or more digits that are not part of a fraction or exponent. The smallest
// unsafe integer, 2^53, has 16 digits, so text without such a run (floats
// like 0.30000000000000004 included) holds no unsafe integer.
const LONG_INTEGER = /(?<![\d.eE+])\d{16,}(?![\d.eE])/;

/** Where a value sits in parsed JSON: object keys and array indexes. */
export type JsonPath = readonly string[];

/** A big integer that parseJsonExactWithIntegers read from a JSON number. */
export interface ExactInteger {
  path: JsonPath;
  digits: string;
}

/**
 * JSON.parse, except integers beyond Number.MAX_SAFE_INTEGER come back as
 * their digit string instead of a rounded number. The backend converts the
 * string to its int when the field is typed as one.
 */
export function parseJsonExact(text: string): unknown {
  return parseJsonExactWithIntegers(text).value;
}

/**
 * parseJsonExact, plus the path of each big integer that was a JSON number,
 * so it can be told apart from a string holding the same digits.
 */
export function parseJsonExactWithIntegers(text: string): {
  value: unknown;
  integers: ExactInteger[];
} {
  if (!LONG_INTEGER.test(text))
    return { value: JSON.parse(text), integers: [] };
  // While JSON.parse runs, a big integer is a string that starts with a
  // control character the text never spells out, so no string in the text
  // can look like one.
  const mark = unusedControlCharacter(text);
  const value = JSON.parse(
    text.replace(JSON_TOKEN, (token) =>
      isUnsafeIntegerText(token) ? `"${mark?.escape ?? ""}${token}"` : token,
    ),
  );
  const integers: ExactInteger[] = [];
  if (!mark) return { value, integers };
  const path: string[] = [];
  const visit = (node: unknown): unknown => {
    if (typeof node === "string") {
      if (!node.startsWith(mark.char)) return node;
      integers.push({ path: [...path], digits: node.slice(1) });
      return node.slice(1);
    }
    if (node && typeof node === "object") {
      const record = node as Record<string, unknown>;
      for (const key of Object.keys(record)) {
        path.push(key);
        record[key] = visit(record[key]);
        path.pop();
      }
    }
    return node;
  };
  return { value: visit(value), integers };
}

// Control characters whose \uXXXX escape is all digits and that have no
// short escape (\b \t \n \f \r), so a string holds one only if the text
// spells out that exact escape.
const MARK_CODES = [
  1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
];

function unusedControlCharacter(
  text: string,
): { char: string; escape: string } | null {
  for (const code of MARK_CODES) {
    const escape = `\\u${code.toString(16).padStart(4, "0")}`;
    if (!text.includes(escape)) {
      return { char: String.fromCharCode(code), escape };
    }
  }
  return null;
}

// A number too large to hold at all (1e400) parses to Infinity, which is
// saved as null: something other than what is on screen.
function hasNonFiniteNumber(text: string): boolean {
  return (text.match(JSON_TOKEN) ?? []).some(
    (token) => !token.startsWith('"') && !Number.isFinite(Number(token)),
  );
}
