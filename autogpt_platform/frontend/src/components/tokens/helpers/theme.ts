import globalsCss from "@/app/globals.css?raw";
import defaultThemeCss from "tailwindcss/theme.css?raw";

const ROOT_FONT_SIZE_PX = 16;

export interface ScaleToken {
  name: string;
  value: string;
  px: number | null;
  isCustom: boolean;
}

// `--{namespace}-{name}: value;` declarations from the @theme blocks of a
// stylesheet, by name.
function themeVariables(css: string, namespace: string) {
  const declaration = new RegExp(`--${namespace}-([\\w.-]+):\\s*([^;]+);`, "g");
  const variables: Record<string, string> = {};
  for (const [, body] of css.matchAll(/@theme[^{]*\{([\s\S]*?)\n\}/g)) {
    for (const [, name, value] of body.matchAll(declaration)) {
      variables[name] = value.replace(/\s+/g, " ").trim();
    }
  }
  return variables;
}

// `--name: value;` declarations of the first top-level block with this
// selector, e.g. `:root` or `.dark`.
function blockVariables(css: string, selector: string) {
  const start = css.indexOf(`\n${selector} {`);
  if (start === -1) return {};
  const body = css.slice(start, css.indexOf("\n}", start));
  const variables: Record<string, string> = {};
  for (const [, name, value] of body.matchAll(/--([\w-]+):\s*([^;]+);/g)) {
    variables[name] = value.trim();
  }
  return variables;
}

function themeVariable(css: string, name: string) {
  return new RegExp(`--${name}:\\s*([^;]+);`).exec(css)?.[1].trim();
}

export function toPx(value: string) {
  const match = /^(-?\d*\.?\d+)(rem|px)?$/.exec(value.trim());
  if (!match) return null;
  const amount = Number(match[1]);
  return match[2] === "rem" ? amount * ROOT_FONT_SIZE_PX : amount;
}

export function formatPx(px: number | null) {
  return px === null ? null : `${Number(px.toFixed(2))}px`;
}

function isSameValue(defaultValue: string | undefined, value: string) {
  if (defaultValue === undefined) return false;
  const defaultPx = toPx(defaultValue);
  return defaultPx === null
    ? defaultValue === value
    : defaultPx === toPx(value);
}

export function toScale(
  scale: Record<string, string>,
  defaults: Record<string, string>,
) {
  const tokens: ScaleToken[] = Object.entries(scale).map(([name, value]) => ({
    name,
    value,
    px: toPx(value),
    isCustom: !isSameValue(defaults[name], value),
  }));

  return tokens.sort(function byPx(a, b) {
    if (a.px === null && b.px === null) return 0;
    if (a.px === null) return 1;
    if (b.px === null) return -1;
    return a.px - b.px;
  });
}

// Tailwind 4 derives every spacing step from one variable, so any multiple
// of 0.25 is a valid step. These are the steps Tailwind documents.
const SPACING_STEPS = [
  0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24,
  28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96,
];

export const SPACING_UNIT =
  themeVariable(globalsCss, "spacing") ??
  themeVariable(defaultThemeCss, "spacing") ??
  "0.25rem";

export function getSpacingScale() {
  const unitPx = toPx(SPACING_UNIT) ?? 4;
  return SPACING_STEPS.map((step) => ({
    name: String(step),
    value: `${step * (unitPx / ROOT_FONT_SIZE_PX)}rem`,
    px: step * unitPx,
  }));
}

// globals.css resets Tailwind's radius scale (`--radius-*: initial`) and
// defines its own, so only its steps exist; Custom marks a step whose value
// differs from Tailwind's default.
export function getBorderRadiusScale() {
  return toScale(
    themeVariables(globalsCss, "radius"),
    themeVariables(defaultThemeCss, "radius"),
  );
}

export interface PaletteFamily {
  name: string;
  steps: { step: string; value: string }[];
}

// The primitive palette from the `@theme static` block, grouped by family.
export function getPalette() {
  const families = new Map<string, PaletteFamily>();
  for (const [name, value] of Object.entries(
    themeVariables(globalsCss, "color"),
  )) {
    const match = /^([a-z]+)-(\d+)$/.exec(name);
    if (!match) continue;
    const family = families.get(match[1]) ?? { name: match[1], steps: [] };
    family.steps.push({ step: match[2], value });
    families.set(match[1], family);
  }
  return [...families.values()];
}

export function getPaletteSingles() {
  const colors = themeVariables(globalsCss, "color");
  return ["white", "black"]
    .filter((name) => name in colors)
    .map((name) => ({ name, value: colors[name] }));
}

// The semantic colours (`bg-background`, `text-muted-foreground`...) with
// the palette step each one points at in :root and in .dark.
export function getSemanticColors() {
  const light = blockVariables(globalsCss, ":root");
  const dark = blockVariables(globalsCss, ".dark");
  const mapped = Object.keys(themeVariables(globalsCss, "color")).filter(
    (name) => name in light,
  );
  return mapped.map((name) => ({
    name,
    light: light[name],
    dark: dark[name],
  }));
}

export function utilityClass(prefix: string, name: string) {
  return name === "DEFAULT" ? prefix : `${prefix}-${name}`;
}
