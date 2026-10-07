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

export function getBorderRadiusScale() {
  const defaults = themeVariables(defaultThemeCss, "radius");
  return toScale(
    { ...defaults, ...themeVariables(globalsCss, "radius") },
    defaults,
  );
}

export function utilityClass(prefix: string, name: string) {
  return name === "DEFAULT" ? prefix : `${prefix}-${name}`;
}
