/**
 * Exports the design tokens that live in code into a form Paper (paper.design) can ingest.
 *
 *   pnpm design:tokens
 *
 * Reads: src/components/styles/colors.ts, src/app/globals.css, tailwind.config.ts,
 *        src/components/atoms/Text/helpers.ts
 * Writes: design/paper/tokens.css  (Tailwind v4 @theme block, for review and diffs)
 *         design/paper/tokens.json (payload for the Paper MCP `create_tokens` tool)
 *
 * Code stays the source of truth; never edit the outputs by hand.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { colors } from "../src/components/styles/colors";
import { variants as textVariants } from "../src/components/atoms/Text/helpers";

type TokenType =
  | "breakpoint"
  | "color"
  | "container"
  | "fontFamily"
  | "fontSize"
  | "fontWeight"
  | "letterSpacing"
  | "lineHeight"
  | "opacity"
  | "radius"
  | "spacing";

interface Token {
  type: TokenType;
  name: string;
  value: string | number;
  description?: string;
}

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = join(root, "design", "paper");
const tokens: Token[] = [];

function add(type: TokenType, name: string, value: string | number, description?: string) {
  tokens.push({ type, name: `--${name}`, value, description });
}

function remToPx(rem: string): string {
  const n = parseFloat(rem);
  return `${Math.round(n * 16 * 100) / 100}px`;
}

function hslToHex(h: number, s: number, l: number): string {
  s /= 100;
  l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = (x: number) => Math.round(x * 255).toString(16).padStart(2, "0");
  return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`.toUpperCase();
}

// ---------- colours ----------
const palette = colors as Record<string, Record<string, string> | string>;
const neutralScales = ["zinc", "slate"];
const otherScales = Object.keys(palette).filter(
  (k) => typeof palette[k] === "object" && !neutralScales.includes(k),
);

function addScale(scale: string) {
  const entries = palette[scale] as Record<string, string>;
  for (const [step, hex] of Object.entries(entries)) {
    add("color", `color-${scale}-${step}`, hex.trim().toUpperCase());
  }
}

neutralScales.forEach(addScale);
otherScales.forEach(addScale);

const globalsCss = readFileSync(join(root, "src/app/globals.css"), "utf8");
const rootBlock = globalsCss.match(/:root\s*{([\s\S]*?)}/)?.[1] ?? "";
const hslVars = new Map<string, string>();
for (const m of rootBlock.matchAll(/--([a-z0-9-]+):\s*([\d.]+)\s+([\d.]+)%\s+([\d.]+)%/g)) {
  hslVars.set(m[1], hslToHex(Number(m[2]), Number(m[3]), Number(m[4])));
}

add("color", "color-white", "#FFFFFF", "Surface: cards, dialogs, inputs");
add("color", "color-black", String(palette.black).toUpperCase(), "Default text colour");
add("color", "color-page-bg", "#F6F7F8", "Body background");
add("color", "color-inset-bg", "#F9F9F9", "Content inset next to the sidebar");
add("color", "color-surface", "var(--color-white)");
add("color", "color-text", "var(--color-black)");
add("color", "color-text-secondary", "var(--color-zinc-600)");
add("color", "color-text-muted", "var(--color-zinc-500)");
add("color", "color-text-grey", String(palette.textGrey).toUpperCase(), "Legacy secondary text");
add("color", "color-border", "var(--color-zinc-200)", "Default border");
add("color", "color-border-card", "var(--color-zinc-300)", "agpt-card border");
add("color", "color-input-border", hslVars.get("input") ?? "#D4D4D8");
add("color", "color-accent", hslVars.get("accent") ?? "#7C3AED", "Selection, focus, emphasis. One moment per screen.");
add("color", "color-accent-soft", "#F5F3FF", "violet-50, used at 50% behind selected cards");
add("color", "color-ring", hslVars.get("ring") ?? "#18181B", "Focus ring");
add("color", "color-button-primary", "var(--color-zinc-800)");
add("color", "color-button-primary-hover", "var(--color-zinc-900)");
add("color", "color-destructive", "var(--color-red-500)");
add("color", "color-success-bg", "#ECFDF5", "emerald-50 (Badge success)");
add("color", "color-success-text", "#047857", "emerald-700");
add("color", "color-warning-bg", "#FFFBEB", "amber-50 (Badge warning)");
add("color", "color-warning-text", "#92400E", "amber-800");
add("color", "color-error-bg", "var(--color-red-50)");
add("color", "color-error-text", "var(--color-red-700)");
add("color", "color-info-bg", "var(--color-zinc-50)");
add("color", "color-info-text", "var(--color-zinc-600)");

// ---------- spacing, radius ----------
const tailwindConfig = readFileSync(join(root, "tailwind.config.ts"), "utf8");
function readBlock(key: string): Array<[string, string]> {
  const block = tailwindConfig.match(new RegExp(`\\n\\s*${key}:\\s*{([\\s\\S]*?)\\n\\s*},`))?.[1] ?? "";
  return [...block.matchAll(/"?([\w.]+)"?:\s*"([^"]+)"/g)].map((m) => [m[1], m[2]]);
}

const spacing = readBlock("spacing")
  .map(([k, v]) => [k, v] as [string, string])
  .sort((a, b) => parseFloat(a[1]) - parseFloat(b[1]));
for (const [k, v] of spacing) {
  add("spacing", `spacing-${k.replace(".", "-")}`, v.endsWith("rem") ? remToPx(v) : v);
}

const radiusNames = readBlock("borderRadius").filter(([, v]) => !v.includes("var("));
for (const [k, v] of radiusNames) {
  add("radius", `radius-${k}`, v.endsWith("rem") ? remToPx(v) : v);
}
add("radius", "radius-md-tw", "6px", "Tailwind rounded-md (badges, toggle buttons)");
add("radius", "radius-2xl-tw", "16px", "Tailwind rounded-2xl (agpt-card)");
add("radius", "radius-3xl-tw", "24px", "Tailwind rounded-3xl (agpt-box)");

// ---------- typography ----------
add("fontFamily", "font-poppins", "Poppins", "Headings h1 to h5");
add("fontFamily", "font-sans", "Geist", "Body and UI");
add("fontFamily", "font-mono", "Geist Mono", "Code");
for (const w of [400, 500, 600, 700]) add("fontWeight", `font-weight-${w}`, w);

for (const [name, cls] of Object.entries(textVariants)) {
  const size = cls.match(/text-\[([\d.]+rem)\]/)?.[1];
  const leading = cls.match(/leading-\[([\d.]+rem)\]/)?.[1];
  const weight = cls.match(/font-\[(\d+)\]/)?.[1] ?? (cls.includes("font-medium") ? "500" : "400");
  const tracking = cls.match(/tracking-\[(-?[\d.]+(?:rem|em))\]/)?.[1];
  if (size) add("fontSize", `text-${name}`, remToPx(size), `Text variant "${name}"`);
  if (leading) add("lineHeight", `leading-${name}`, remToPx(leading));
  add("fontWeight", `font-weight-${name}`, Number(weight));
  if (tracking) {
    add("letterSpacing", `tracking-${name}`, tracking.endsWith("rem") ? remToPx(tracking) : tracking);
  }
}

// ---------- layout ----------
for (const [k, v] of Object.entries({ sm: 640, md: 768, lg: 1024, xl: 1280, "2xl": 1536 })) {
  add("breakpoint", `breakpoint-${k}`, `${v}px`);
}
add("breakpoint", "breakpoint-mobile-design", "390px", "Design width for mobile artboards");
add("breakpoint", "breakpoint-desktop-design", "1440px", "Design width for desktop artboards");
add("container", "container-7xl", "1280px", "max-w-7xl: header and content width");
add("container", "container-page", "1400px", "Tailwind container 2xl");
add("container", "container-sidebar", "292px", "--sidebar-width 18.25rem");
add("container", "container-sidebar-tour", "304px", "Tour sidebar 19rem");
add("opacity", "opacity-disabled", 0.5);
add("opacity", "opacity-loading", 0.6);

// ---------- write ----------
mkdirSync(outDir, { recursive: true });
const css = [
  "/* Generated by scripts/export-design-tokens.ts — do not edit. Run `pnpm design:tokens`. */",
  "@theme {",
  ...tokens.map((t) => `  ${t.name}: ${typeof t.value === "number" ? t.value : t.value};${t.description ? ` /* ${t.description} */` : ""}`),
  "}",
  "",
].join("\n");
writeFileSync(join(outDir, "tokens.css"), css);
writeFileSync(join(outDir, "tokens.json"), JSON.stringify(tokens, null, 2) + "\n");
console.log(`wrote ${tokens.length} tokens to design/paper/tokens.{css,json}`);
