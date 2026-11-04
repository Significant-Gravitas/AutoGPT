/**
 * The colour utilities the colour codemods (alias-to-role.ts,
 * palette-to-role.ts) and design-system/no-alias-colours and
 * no-palette-colours read, grouped by how a class uses its colour.
 */

/** Text and icon colours. */
export const TEXT_UTILITIES = [
  "text",
  "fill",
  "stroke",
  "decoration",
  "caret",
  "placeholder",
  "accent",
];

/** Background fills and gradient stops. */
export const FILL_UTILITIES = ["bg", "from", "via", "to"];

/** Borders, rings, dividers, outlines and shadows. */
export const LINE_UTILITIES = [
  "border",
  "border-x",
  "border-y",
  "border-t",
  "border-r",
  "border-b",
  "border-l",
  "border-s",
  "border-e",
  "ring",
  "ring-offset",
  "divide",
  "outline",
  "shadow",
];

export const COLOUR_UTILITIES = [
  ...TEXT_UTILITIES,
  ...FILL_UTILITIES,
  ...LINE_UTILITIES,
];

/**
 * `{ "border": "border-base" }` -> every `<utility>-border` class, keyed to
 * `<utility>-border-base`, for the given utilities.
 */
export function expandNames(
  names: Record<string, string>,
  utilities: string[] = COLOUR_UTILITIES,
) {
  const map = new Map<string, string>();
  for (const [name, role] of Object.entries(names)) {
    for (const utility of utilities) {
      map.set(`${utility}-${name}`, `${utility}-${role}`);
    }
  }
  return map;
}

// A class as one regex: variants, optional `!`, the utility and its value,
// then an optional /opacity. Used to find colour classes in any string.
const UTILITY_GROUP = [...COLOUR_UTILITIES]
  .sort((a, b) => b.length - a.length)
  .join("|");

export function colourClassPattern(values: string) {
  return new RegExp(
    String.raw`(?<![\w\-/\[.#])((?:[^\s"'\`{}]*:)?!?)(${UTILITY_GROUP})-(${values})(\/(?:\d+(?:\.\d+)?|\[[^\]\s]+\]))?(?![\w\-])`,
    "g",
  );
}
