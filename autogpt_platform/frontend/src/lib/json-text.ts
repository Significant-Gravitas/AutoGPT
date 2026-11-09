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
  if (hasInexactNumber(text)) return false;
  try {
    return JSON.stringify(JSON.parse(text)) === JSON.stringify(value);
  } catch {
    return false;
  }
}

const STRING_LITERAL = /"(?:[^"\\]|\\.)*"/g;
const NUMBER_LITERAL = /-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;

// A number JSON.parse can only round (an integer that doesn't read back as
// the same digits, or one too large to hold at all) would be saved as
// something other than what is on screen. An integer written as it is stored
// reads back unchanged, so the field stops being rewritten once it shows it.
function hasInexactNumber(text: string): boolean {
  const numbers = text.replace(STRING_LITERAL, '""').match(NUMBER_LITERAL);
  return (numbers ?? []).some((literal) => {
    const parsed = Number(literal);
    if (!Number.isFinite(parsed)) return true;
    // -0 is exact too (it is on the way to "-0.5"), but reads back as "0".
    if (!/^-?\d+$/.test(literal) || Object.is(parsed, -0)) return false;
    return String(parsed) !== literal;
  });
}
