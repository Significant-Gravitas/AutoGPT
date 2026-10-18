/**
 * True when `text` is JSON for `value`, however it is spaced or indented.
 * A field that edits JSON as text keeps the user's own text while it still
 * says the same thing as the stored value. Replacing it with a reformatted
 * copy rewrites the field under the user, which throws the caret to the end
 * and cancels an IME composition.
 */
export function isJsonTextFor(text: string, value: unknown): boolean {
  if (text.trim() === "") return value === undefined || value === null;
  try {
    return JSON.stringify(JSON.parse(text)) === JSON.stringify(value);
  } catch {
    return false;
  }
}
