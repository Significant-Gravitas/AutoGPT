// @vitest-environment node
import { describe, expect, it } from "vitest";
import { contrastRatio, type Rgb } from "../../../scripts/ds/canary-report";
import { defaultTokens } from "./token-css";

// WCAG contrast of the colour pairs that carry text, in every theme, computed
// from the role triplets. 4.5:1 for text, a placeholder's included (WCAG
// 1.4.3). See docs/engineering/design-system.md.

const TEXT = 4.5;

const BODY_TEXT = [
  "foreground-base",
  "foreground-strong",
  "foreground-emphasis",
  "foreground-body",
  "foreground-secondary",
  "foreground-tertiary",
];
const PAGE = ["canvas", "surface", "surface-muted"];

const PAIRS: [text: string, background: string, minimum: number][] = [
  ...BODY_TEXT.flatMap((text) =>
    PAGE.map((page) => [text, page, TEXT] as [string, string, number]),
  ),
  ["action-foreground", "action", TEXT],
  ["foreground-inverse", "surface-inverse", TEXT],
  ["error-inverse", "surface-inverse", TEXT],
  ["success-inverse", "surface-inverse", TEXT],
  ["accent-strong", "accent-surface", TEXT],
  ["error-foreground", "error-surface", TEXT],
  ["warning-foreground", "warning-surface", TEXT],
  ["success-foreground", "success-surface", TEXT],
  ["info-foreground", "info-surface", TEXT],
  ["placeholder", "surface", TEXT],
];

// Pairs the light theme fails today, with the ratio measured. They are left
// as they are until design picks new values; a change to either colour moves
// the ratio and fails here, so the list only ever shrinks deliberately.
const LIGHT_BELOW_MINIMUM: Record<string, string> = {
  "foreground-tertiary on canvas": "3.50",
  "foreground-tertiary on surface": "3.72",
  "foreground-tertiary on surface-muted": "3.57",
  "error-inverse on surface-inverse": "3.54",
  "placeholder on surface": "3.72",
};

function palette(tokens: Map<string, string>) {
  return (role: string): Rgb => {
    const value = tokens.get(`--color-${role}`);
    if (!value) throw new Error(`no --color-${role}`);
    return value.split(" ").map(Number) as Rgb;
  };
}

const THEMES = [
  {
    name: "light",
    colour: palette(defaultTokens()),
    allowed: LIGHT_BELOW_MINIMUM,
  },
];

describe.each(THEMES)("$name theme contrast", ({ colour, allowed }) => {
  it.each(PAIRS)("%s on %s", (text, background, minimum) => {
    const ratio = contrastRatio(colour(text), colour(background));
    const pair = `${text} on ${background}`;
    if (pair in allowed) {
      expect(ratio).toBeLessThan(minimum);
      expect(ratio.toFixed(2)).toBe(allowed[pair]);
    } else {
      expect(ratio, `${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(minimum);
    }
  });

  it("allows only pairs it checks", () => {
    const checked = PAIRS.map(
      ([text, background]) => `${text} on ${background}`,
    );
    expect(
      Object.keys(allowed).filter((pair) => !checked.includes(pair)),
    ).toEqual([]);
  });
});
