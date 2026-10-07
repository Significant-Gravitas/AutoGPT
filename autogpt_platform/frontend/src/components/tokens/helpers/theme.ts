import defaultTheme from "tailwindcss/defaultTheme";
import resolveConfig from "tailwindcss/resolveConfig";
import tailwindConfig from "../../../../tailwind.config";

const ROOT_FONT_SIZE_PX = 16;

export const resolvedTheme = resolveConfig(tailwindConfig).theme;

export interface ScaleToken {
  name: string;
  value: string;
  px: number | null;
  isCustom: boolean;
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

export function getSpacingScale() {
  return toScale(
    resolvedTheme.spacing as Record<string, string>,
    defaultTheme.spacing as Record<string, string>,
  );
}

export function getBorderRadiusScale() {
  return toScale(
    resolvedTheme.borderRadius as Record<string, string>,
    defaultTheme.borderRadius as Record<string, string>,
  );
}

export function utilityClass(prefix: string, name: string) {
  return name === "DEFAULT" ? prefix : `${prefix}-${name}`;
}
