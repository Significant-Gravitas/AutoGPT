/**
 * The analysis half of scripts/ds/canary-check.ts: what the canary look's
 * values are, which computed values are leaks, and the report. The browser
 * half only collects computed styles; everything here is pure so it can be
 * tested (scripts/__tests__/canary-report.test.ts).
 */

export type Rgb = [number, number, number];

export interface ParsedColour {
  rgb: Rgb;
  alpha: number;
}

/** What the page script collects for one element (or pseudo-element). */
export interface ElementFacts {
  element: string;
  within: string | null;
  origin: string;
  colours: { property: string; value: string }[];
  radius: { corners: string[]; width: number; height: number } | null;
  font: string | null;
}

export type LeakKind = "colour" | "radius" | "font";

export interface Leak {
  kind: LeakKind;
  property: string;
  value: string;
  element: string;
  within: string | null;
  origin: string;
}

export interface Canary {
  palette: Rgb[];
  fonts: string[];
}

/** The colour triplets and font families canary.css sets. */
export function readCanary(css: string): Canary {
  const body = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const palette = [
    ...body.matchAll(
      /--(?:color-[a-z-]+|shadow-color):\s*(\d+)\s+(\d+)\s+(\d+)\s*;/g,
    ),
  ].map(([, r, g, b]) => [Number(r), Number(g), Number(b)] as Rgb);
  const fonts = [...body.matchAll(/--font-[a-z-]+:\s*([^;]+);/g)].map(
    ([, value]) => firstFamily(value),
  );
  return { palette, fonts: [...new Set(fonts)] };
}

export function firstFamily(fontFamily: string) {
  return fontFamily
    .split(",")[0]
    .trim()
    .replace(/^["']|["']$/g, "");
}

const COLOUR_FUNCTION = /(rgba?|color|oklab|oklch|lab|lch|hsla?)\(([^()]*)\)/gi;

function channel(text: string, scale: number) {
  const value = parseFloat(text);
  return text.trim().endsWith("%") ? (value / 100) * scale : value;
}

function linearToSrgb(value: number) {
  const v =
    value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
  return Math.round(Math.min(1, Math.max(0, v)) * 255);
}

function oklabToRgb(l: number, a: number, b: number): Rgb {
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    linearToSrgb(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_),
    linearToSrgb(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_),
    linearToSrgb(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_),
  ];
}

/**
 * Every colour in a computed value (`rgba(0, 0, 0, 0.1) 0px 1px 2px`, a
 * gradient, `color(srgb 1 0 0 / 0.5)`). A colour space it cannot convert
 * comes back as null, so the caller reports it as is.
 */
export function coloursIn(value: string) {
  const found: { text: string; colour: ParsedColour | null }[] = [];
  for (const match of value.matchAll(COLOUR_FUNCTION)) {
    const [text, fn, args] = match;
    const [main, slashAlpha] = args.split("/");
    const parts = main
      .trim()
      .split(/[\s,]+/)
      .filter(Boolean);
    const name = fn.toLowerCase();
    let colour: ParsedColour | null = null;
    if (name === "rgb" || name === "rgba") {
      const [r, g, b, commaAlpha] = parts;
      colour = {
        rgb: [channel(r, 255), channel(g, 255), channel(b, 255)],
        alpha: channel(slashAlpha ?? commaAlpha ?? "1", 1),
      };
    } else if (name === "color" && parts[0] === "srgb") {
      colour = {
        rgb: parts.slice(1, 4).map((part) => channel(part, 1) * 255) as Rgb,
        alpha: channel(slashAlpha ?? "1", 1),
      };
    } else if (name === "oklab" || name === "oklch") {
      const l = channel(parts[0], 1);
      let [a, b] = [channel(parts[1], 0.4), channel(parts[2], 0.4)];
      if (name === "oklch") {
        const hue = (parseFloat(parts[2]) * Math.PI) / 180;
        [a, b] = [a * Math.cos(hue), a * Math.sin(hue)];
      }
      colour = {
        rgb: oklabToRgb(l, a, b),
        alpha: channel(slashAlpha ?? "1", 1),
      };
    }
    found.push({ text, colour });
  }
  return found;
}

// Rounding in color-mix() and oklab conversions moves a channel by a step or two.
const TOLERANCE = 2.5;

export function isCanaryColour(colour: ParsedColour, palette: Rgb[]) {
  if (colour.alpha === 0) return true;
  return palette.some((swatch) =>
    swatch.every(
      (value, index) => Math.abs(value - colour.rgb[index]) <= TOLERANCE,
    ),
  );
}

/** The colour leaks in one element's computed colours. */
// Splits a computed shadow list at its top-level commas.
function shadowLayers(value: string) {
  const layers: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < value.length; i++) {
    if (value[i] === "(") depth++;
    else if (value[i] === ")") depth--;
    else if (value[i] === "," && depth === 0) {
      layers.push(value.slice(start, i));
      start = i + 1;
    }
  }
  return [...layers, value.slice(start)];
}

/**
 * A shadow without the layers that draw nothing: Tailwind's ring and
 * ring-offset placeholders (`rgb(255, 255, 255) 0px 0px 0px 0px`) carry a
 * colour but no size.
 */
export function visibleShadow(value: string) {
  return shadowLayers(value)
    .filter((layer) =>
      (layer.replace(COLOUR_FUNCTION, "").match(/-?[\d.]+px/g) ?? []).some(
        (length) => parseFloat(length) !== 0,
      ),
    )
    .join(",");
}

export function colourLeaks(facts: ElementFacts, canary: Canary): Leak[] {
  const leaks: Leak[] = [];
  for (const { property, value: raw } of facts.colours) {
    const value = /shadow$/.test(property) ? visibleShadow(raw) : raw;
    for (const { text, colour } of coloursIn(value)) {
      if (colour && isCanaryColour(colour, canary.palette)) continue;
      leaks.push({
        kind: "colour",
        property,
        value: text,
        element: facts.element,
        within: facts.within,
        origin: facts.origin,
      });
    }
  }
  return leaks;
}

// A corner is "8px", "50%" or an elliptical "8px 4px": across, then down. A
// percentage is of the width across and of the height down.
function cornerAxes(corner: string, width: number, height: number) {
  const [across, down = across] = corner.trim().split(/\s+/);
  const px = (part: string, size: number) =>
    part.endsWith("%")
      ? (parseFloat(part) / 100) * size
      : parseFloat(part) || 0;
  return [px(across, width), px(down, height)];
}

/**
 * A radius leak: a rounded corner on anything but a circle. `--radius-full`
 * keeps its value in the canary look, so a pill (a full radius on a box that
 * is not square) is reported apart, as `border-radius (full)`.
 */
export function radiusLeak(facts: ElementFacts): Leak | null {
  if (!facts.radius) return null;
  const { corners, width, height } = facts.radius;
  const px = corners.flatMap((corner) => cornerAxes(corner, width, height));
  if (px.every((value) => value === 0)) return null;
  // A circle is a square whose corners reach half its side on both axes. A
  // box of unknown size (0) is not one.
  const side = Math.min(width, height);
  const square = side > 0 && Math.abs(width - height) <= 1;
  if (square && px.every((value) => value >= side / 2 - 0.5)) return null;
  const full = px.some((value) => value >= 1000);
  return {
    kind: "radius",
    property: full ? "border-radius (full)" : "border-radius",
    value: [...new Set(corners)].join(" / "),
    element: facts.element,
    within: facts.within,
    origin: facts.origin,
  };
}

export function fontLeak(facts: ElementFacts, canary: Canary): Leak | null {
  if (!facts.font || canary.fonts.includes(firstFamily(facts.font)))
    return null;
  return {
    kind: "font",
    property: "font-family",
    value: facts.font,
    element: facts.element,
    within: facts.within,
    origin: facts.origin,
  };
}

export function leaksOf(facts: ElementFacts, canary: Canary) {
  return [
    ...colourLeaks(facts, canary),
    radiusLeak(facts),
    fontLeak(facts, canary),
  ].filter((leak): leak is Leak => leak !== null);
}

/** WCAG 2 relative luminance of an sRGB colour. */
function relativeLuminance([r, g, b]: Rgb) {
  const linear = (value: number) => {
    const c = value / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
}

/** WCAG 2 contrast ratio, from 1 to 21. */
export function contrastRatio(a: Rgb, b: Rgb) {
  const [high, low] = [relativeLuminance(a), relativeLuminance(b)].sort(
    (x, y) => y - x,
  );
  return (high + 0.05) / (low + 0.05);
}

export interface GroupedLeak extends Leak {
  count: number;
  targets: string[];
}

export interface Report {
  totals: Record<LeakKind, { instances: number; distinct: number }>;
  leaks: Record<LeakKind, GroupedLeak[]>;
  byValue: Record<LeakKind, { value: string; count: number }[]>;
}

/**
 * De-duplicates leaks by kind, property, value, element and origin, with
 * counts.
 */
export function buildReport(
  found: { target: string; leaks: Leak[] }[],
): Report {
  const groups = new Map<string, GroupedLeak>();
  for (const { target, leaks } of found) {
    for (const leak of leaks) {
      const key = [
        leak.kind,
        leak.property,
        leak.value,
        leak.element,
        leak.within,
        leak.origin,
      ].join("\u0000");
      const group = groups.get(key) ?? { ...leak, count: 0, targets: [] };
      group.count++;
      if (!group.targets.includes(target)) group.targets.push(target);
      groups.set(key, group);
    }
  }
  const kinds: LeakKind[] = ["colour", "radius", "font"];
  const all = [...groups.values()].sort((a, b) => b.count - a.count);
  const ofKind = (kind: LeakKind) => all.filter((leak) => leak.kind === kind);
  return {
    totals: Object.fromEntries(
      kinds.map((kind) => [
        kind,
        {
          instances: ofKind(kind).reduce((sum, leak) => sum + leak.count, 0),
          distinct: ofKind(kind).length,
        },
      ]),
    ) as Report["totals"],
    leaks: Object.fromEntries(
      kinds.map((kind) => [kind, ofKind(kind)]),
    ) as Report["leaks"],
    byValue: Object.fromEntries(
      kinds.map((kind) => {
        const counts = new Map<string, number>();
        for (const leak of ofKind(kind))
          counts.set(leak.value, (counts.get(leak.value) ?? 0) + leak.count);
        return [
          kind,
          [...counts]
            .map(([value, count]) => ({ value, count }))
            .sort((a, b) => b.count - a.count),
        ];
      }),
    ) as Report["byValue"],
  };
}

function targetList(targets: string[]) {
  const shown = targets.slice(0, 3).join(", ");
  return targets.length > 3 ? `${shown}, +${targets.length - 3} more` : shown;
}

/** The report as text, the top `limit` rows of each kind. */
export function formatReport(report: Report, limit = 60) {
  const lines: string[] = [];
  for (const kind of Object.keys(report.leaks) as LeakKind[]) {
    const { instances, distinct } = report.totals[kind];
    lines.push(
      `## ${kind} leaks: ${instances} on ${distinct} distinct element/value pairs`,
      "",
    );
    if (distinct === 0) continue;
    lines.push("Most common values:");
    for (const { value, count } of report.byValue[kind].slice(0, 15))
      lines.push(`  ${String(count).padStart(5)}x  ${value}`);
    lines.push("", "By element:");
    for (const leak of report.leaks[kind].slice(0, limit)) {
      const within = leak.within ? ` (in ${leak.within})` : "";
      lines.push(
        `  ${String(leak.count).padStart(5)}x  ${leak.property}: ${leak.value}  ${leak.element}${within} [${leak.origin}]  ${targetList(leak.targets)}`,
      );
    }
    if (report.leaks[kind].length > limit)
      lines.push(
        `  ... ${report.leaks[kind].length - limit} more in the JSON report`,
      );
    lines.push("");
  }
  return lines.join("\n");
}
