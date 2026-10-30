/**
 * Swaps raw palette and arbitrary typography classes for the design tokens
 * in src/app/globals.css (SECRT-3099, DS1).
 *
 * Two kinds of rewrite:
 * - exact (TOKEN_GROUPS, SIZE_TOKENS): `scripts/check-token-equivalence.ts`
 *   proves each one renders the same computed value;
 * - snapped (SNAP_GROUPS, SNAP_SIZE_TOKENS): the role with the nearest value,
 *   per the SECRT-3097 plan (5.1 greys, 5.2 hues, 5.6 type). These move
 *   pixels a little, so the equivalence check leaves them out. Rows the plan
 *   flagged as noticeably different (a new hue, a much paler fill) sit in
 *   FLAGGED_SNAP_GROUPS and only run with `--flagged`.
 * Anything else is left alone and stays at `warn` in the lint until someone
 * defines a token for it.
 *
 *   npx tsx scripts/codemod-tokens.ts <file|dir>...            # dry run
 *   npx tsx scripts/codemod-tokens.ts --write <file|dir>...    # rewrite files
 *   npx tsx scripts/codemod-tokens.ts --exact ...              # exact rows only
 *   npx tsx scripts/codemod-tokens.ts --flagged ...            # + flagged rows
 *
 * Variant prefixes (`hover:`, `md:`, `data-[state=open]:`), `!important` and
 * opacity modifiers (`/50`) are kept as they are.
 *
 * Size tokens bundle a line-height, while a raw size such as `text-[13px]`
 * sets none. Within one class list, a size swap therefore:
 * - keeps an explicit `leading-*` that follows it in the same variant scope
 *   (it wins over the bundle in CSS and in cn());
 * - otherwise appends `leading-inherit`, which renders exactly like the
 *   line-height the raw size never set;
 * - is skipped, and reported, when a `leading-*` comes before it or sits in
 *   another variant scope, or when the size is `!important` next to any
 *   `leading-*`: plain CSS and cn() resolve those differently, so no single
 *   rewrite is exact.
 *
 * A snapped token that carries its own opacity (`bg-error/20`) is skipped,
 * and reported, on a class that already has one (`bg-red-200/50`).
 *
 * Directory PRs run this script; they do not edit it.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, relative } from "node:path";

export interface TokenMapping {
  /** The token class, e.g. `bg-surface`. */
  token: string;
  /** Classes this token replaces in the codemod. Each class maps once. */
  replaces: string[];
}

export interface TokenGroup {
  group: string;
  kind: "color" | "typography";
  mappings: TokenMapping[];
}

/** One-for-one swaps: colours, line-heights, letter-spacing, weights. */
export const TOKEN_GROUPS: TokenGroup[] = [
  {
    group: "Surfaces",
    kind: "color",
    mappings: [
      { token: "bg-canvas", replaces: ["bg-[#F6F7F8]", "bg-[#f6f7f8]"] },
      { token: "bg-surface", replaces: ["bg-white"] },
      { token: "bg-surface-muted", replaces: ["bg-zinc-50"] },
      { token: "bg-surface-hover", replaces: ["bg-zinc-100"] },
      { token: "bg-surface-active", replaces: ["bg-zinc-200"] },
      { token: "bg-overlay", replaces: ["bg-stone-500"] },
    ],
  },
  {
    group: "Text",
    kind: "color",
    mappings: [
      { token: "text-foreground-base", replaces: ["text-black"] },
      { token: "text-foreground-strong", replaces: ["text-zinc-900"] },
      { token: "text-foreground-emphasis", replaces: ["text-zinc-800"] },
      {
        token: "text-foreground-body",
        replaces: ["text-zinc-700", "text-[#505057]"],
      },
      { token: "text-foreground-secondary", replaces: ["text-zinc-600"] },
      {
        token: "text-foreground-tertiary",
        replaces: ["text-zinc-500", "text-[#83838C]", "text-[#83838c]"],
      },
      { token: "text-foreground-subtle", replaces: ["text-zinc-400"] },
      { token: "text-foreground-disabled", replaces: ["text-zinc-300"] },
      {
        token: "placeholder:text-placeholder",
        replaces: ["placeholder:text-zinc-500"],
      },
    ],
  },
  {
    group: "Borders",
    kind: "color",
    mappings: [
      {
        token: "border-separator",
        replaces: ["border-zinc-100"],
      },
      { token: "divide-separator", replaces: ["divide-zinc-100"] },
      { token: "border-border-base", replaces: ["border-zinc-200"] },
      { token: "divide-border-base", replaces: ["divide-zinc-200"] },
      { token: "ring-border-base", replaces: ["ring-zinc-200"] },
      { token: "border-border-strong", replaces: ["border-zinc-300"] },
      { token: "ring-border-strong", replaces: ["ring-zinc-300"] },
      { token: "ring-focus", replaces: ["ring-neutral-950"] },
    ],
  },
  {
    group: "Action",
    kind: "color",
    mappings: [
      { token: "bg-action", replaces: ["bg-zinc-800"] },
      { token: "border-action", replaces: ["border-zinc-800"] },
      { token: "bg-action-hover", replaces: ["bg-zinc-900"] },
      { token: "border-action-hover", replaces: ["border-zinc-900"] },
      { token: "text-action-foreground", replaces: ["text-white"] },
    ],
  },
  {
    group: "Accent",
    kind: "color",
    mappings: [
      { token: "text-brand", replaces: ["text-purple-500"] },
      { token: "bg-brand", replaces: ["bg-purple-500"] },
      { token: "bg-brand-hover", replaces: ["bg-purple-600"] },
      { token: "bg-accent-surface", replaces: ["bg-violet-50"] },
      { token: "text-accent-strong", replaces: ["text-violet-700"] },
    ],
  },
  {
    group: "Status",
    kind: "color",
    mappings: [
      { token: "text-error", replaces: ["text-red-500"] },
      { token: "bg-error", replaces: ["bg-red-500"] },
      { token: "border-error", replaces: ["border-red-500"] },
      { token: "bg-error-hover", replaces: ["bg-red-600"] },
      { token: "border-error-hover", replaces: ["border-red-600"] },
      { token: "text-error-foreground", replaces: ["text-red-700"] },
      { token: "bg-error-surface", replaces: ["bg-red-50"] },
      { token: "text-warning", replaces: ["text-amber-500"] },
      { token: "bg-warning", replaces: ["bg-amber-500"] },
      { token: "text-warning-foreground", replaces: ["text-amber-700"] },
      { token: "bg-warning-surface", replaces: ["bg-amber-50"] },
      { token: "border-warning-border", replaces: ["border-amber-200"] },
      { token: "ring-warning-border", replaces: ["ring-amber-200"] },
      { token: "text-success", replaces: ["text-green-600"] },
      { token: "text-success-foreground", replaces: ["text-green-700"] },
      { token: "bg-success-surface", replaces: ["bg-green-50"] },
      { token: "text-info", replaces: ["text-blue-500"] },
      { token: "bg-info", replaces: ["bg-blue-500"] },
      { token: "text-info-foreground", replaces: ["text-blue-600"] },
      { token: "bg-info-surface", replaces: ["bg-blue-50"] },
    ],
  },
  {
    group: "Line height",
    kind: "typography",
    mappings: [
      { token: "leading-4", replaces: ["leading-[16px]", "leading-[1rem]"] },
      {
        token: "leading-4.5",
        replaces: ["leading-[18px]", "leading-[1.125rem]"],
      },
      { token: "leading-5", replaces: ["leading-[20px]", "leading-[1.25rem]"] },
      {
        token: "leading-5.5",
        replaces: ["leading-[22px]", "leading-[1.375rem]"],
      },
      { token: "leading-6", replaces: ["leading-[24px]", "leading-[1.5rem]"] },
      {
        token: "leading-6.5",
        replaces: ["leading-[26px]", "leading-[1.625rem]"],
      },
      { token: "leading-7", replaces: ["leading-[28px]", "leading-[1.75rem]"] },
      { token: "leading-8", replaces: ["leading-[32px]", "leading-[2rem]"] },
      { token: "leading-10", replaces: ["leading-[40px]", "leading-[2.5rem]"] },
    ],
  },
  {
    group: "Letter spacing",
    kind: "typography",
    mappings: [
      { token: "tracking-display", replaces: ["tracking-[-0.04em]"] },
      { token: "tracking-heading", replaces: ["tracking-[-0.02em]"] },
      { token: "tracking-title", replaces: ["tracking-[-0.01em]"] },
      { token: "tracking-caps", replaces: ["tracking-[0.12em]"] },
      { token: "tracking-eyebrow", replaces: ["tracking-[0.06em]"] },
    ],
  },
  {
    group: "Font weight",
    kind: "typography",
    mappings: [
      { token: "font-normal", replaces: ["font-[400]"] },
      { token: "font-medium", replaces: ["font-[500]"] },
      { token: "font-semibold", replaces: ["font-[600]"] },
    ],
  },
];

/** Size roles. Each bundles a line-height; see the header for the rules. */
export const SIZE_TOKENS: TokenMapping[] = [
  { token: "text-tiny", replaces: ["text-[10px]", "text-[0.625rem]"] },
  { token: "text-caption-sm", replaces: ["text-[11px]", "text-[0.6875rem]"] },
  { token: "text-caption", replaces: ["text-[12px]", "text-[0.75rem]"] },
  { token: "text-body-sm", replaces: ["text-[13px]", "text-[0.8125rem]"] },
  { token: "text-body", replaces: ["text-[14px]", "text-[0.875rem]"] },
  { token: "text-body-md", replaces: ["text-[15px]", "text-[0.9375rem]"] },
  { token: "text-body-lg", replaces: ["text-[16px]", "text-[1rem]"] },
  { token: "text-lead", replaces: ["text-[20px]", "text-[1.25rem]"] },
  { token: "text-heading-sm", replaces: ["text-[22px]", "text-[1.375rem]"] },
  { token: "text-heading", replaces: ["text-[28px]", "text-[1.75rem]"] },
  { token: "text-title", replaces: ["text-[32px]", "text-[2rem]"] },
  { token: "text-display", replaces: ["text-[44px]", "text-[2.75rem]"] },
];

/** Colour utilities a snapped role is written into, by how the class uses it. */
const TEXT_UTILITIES = ["text", "fill", "stroke", "decoration", "caret"];
const FILL_UTILITIES = ["bg", "from", "via", "to"];
const LINE_UTILITIES = [
  "border",
  "border-x",
  "border-y",
  "border-t",
  "border-r",
  "border-b",
  "border-l",
  "ring",
  "ring-offset",
  "divide",
  "outline",
  "shadow",
];

/**
 * One palette step and its nearest role: `text` as text or an icon colour,
 * `fill` as a background, `line` as a border, ring, divider or outline. A
 * use the row leaves out is not rewritten, e.g. `bg-gray-500`.
 */
interface ColorSnap {
  step: string;
  text?: string;
  fill?: string;
  line?: string;
}

function everywhere(step: string, role: string): ColorSnap {
  return { step, text: role, fill: role, line: role };
}

function snapGroup(group: string, snaps: ColorSnap[]): TokenGroup {
  const mappings: TokenMapping[] = [];
  const add = (utilities: string[], step: string, role?: string) => {
    if (!role) return;
    for (const utility of utilities) {
      mappings.push({
        token: `${utility}-${role}`,
        replaces: [`${utility}-${step}`],
      });
    }
  };
  for (const { step, text, fill, line } of snaps) {
    add(TEXT_UTILITIES, step, text);
    add(FILL_UTILITIES, step, fill);
    add(LINE_UTILITIES, step, line);
    if (text) {
      const placeholder = text === "foreground-tertiary" ? "placeholder" : text;
      mappings.push({
        token: `placeholder:text-${placeholder}`,
        replaces: [`placeholder:text-${step}`],
      });
    }
  }
  return { group, kind: "color", mappings };
}

/** Plan 5.1: the grey ramps onto the neutral roles. */
const GREY_SNAPS: ColorSnap[] = [
  ...["gray", "neutral", "slate"].flatMap((ramp): ColorSnap[] => [
    { step: `${ramp}-50`, fill: "surface-muted", line: "surface-muted" },
    { step: `${ramp}-100`, fill: "surface-hover", line: "separator" },
    { step: `${ramp}-200`, fill: "surface-active", line: "border-base" },
    { step: `${ramp}-300`, fill: "border-strong", line: "border-strong" },
    { step: `${ramp}-500`, text: "foreground-tertiary" },
    { step: `${ramp}-600`, text: "foreground-secondary" },
    { step: `${ramp}-700`, text: "foreground-body" },
  ]),
  { step: "gray-400", text: "foreground-subtle", line: "border-strong" },
  { step: "neutral-400", text: "foreground-subtle" },
  { step: "slate-400", text: "foreground-subtle" },
  {
    step: "gray-800",
    text: "foreground-emphasis",
    fill: "action",
    line: "action",
  },
  {
    step: "gray-900",
    text: "foreground-base",
    fill: "action-hover",
    line: "action-hover",
  },
  {
    step: "neutral-800",
    text: "foreground-emphasis",
    fill: "action",
    line: "action",
  },
  {
    step: "neutral-900",
    text: "foreground-strong",
    fill: "action-hover",
    line: "action-hover",
  },
  { step: "neutral-950", text: "foreground-base", fill: "focus" },
  { step: "neutral-50", text: "action-foreground" },
  // Inside build/, slate-200 as a node ring is `node-border` (same hex):
  // swap those by hand.
  { step: "slate-800", text: "foreground-body" },
  { step: "slate-900", text: "foreground-strong" },
  { step: "slate-950", text: "foreground-base" },
  // zinc is the roles' own ramp, so these are zero-pixel except zinc-950.
  { step: "zinc-50", fill: "surface-muted", line: "surface-muted" },
  { step: "zinc-100", fill: "surface-hover", line: "separator" },
  { step: "zinc-200", fill: "surface-active", line: "border-base" },
  {
    step: "zinc-300",
    text: "foreground-disabled",
    fill: "border-strong",
    line: "border-strong",
  },
  everywhere("zinc-400", "foreground-subtle"),
  everywhere("zinc-500", "foreground-tertiary"),
  everywhere("zinc-600", "foreground-secondary"),
  everywhere("zinc-700", "foreground-body"),
  {
    step: "zinc-800",
    text: "foreground-emphasis",
    fill: "action",
    line: "action",
  },
  {
    step: "zinc-900",
    text: "foreground-strong",
    fill: "action-hover",
    line: "action-hover",
  },
  { step: "zinc-950", text: "foreground-base" },
  { step: "stone-400", text: "foreground-subtle" },
  { step: "stone-800", text: "foreground-emphasis" },
  {
    step: "white",
    text: "action-foreground",
    fill: "surface",
    line: "surface",
  },
  { step: "black", text: "foreground-base" },
];

/** Plan 5.2: status and accent hues, the rows not flagged. */
const HUE_SNAPS: ColorSnap[] = [
  {
    step: "red-600",
    text: "error-foreground",
    fill: "error-hover",
    line: "error-hover",
  },
  everywhere("red-400", "error"),
  everywhere("red-500", "error"),
  { step: "red-50", fill: "error-surface" },
  { step: "red-100", fill: "error-surface" },
  { step: "red-200", fill: "error/20", line: "error/20" },
  { step: "red-300", fill: "error/40", line: "error/40" },
  { step: "red-700", text: "error-foreground" },
  { step: "red-800", text: "error-foreground" },
  { step: "amber-600", text: "warning-foreground" },
  { step: "amber-700", text: "warning-foreground" },
  { step: "amber-800", text: "warning-foreground" },
  { step: "amber-900", text: "warning-foreground" },
  everywhere("amber-400", "warning"),
  everywhere("amber-500", "warning"),
  { step: "amber-50", fill: "warning-surface" },
  { step: "amber-100", fill: "warning-surface" },
  { step: "amber-200", fill: "warning-border", line: "warning-border" },
  { step: "amber-300", fill: "warning-border", line: "warning-border" },
  { step: "yellow-50", fill: "warning-surface" },
  { step: "yellow-100", fill: "warning-surface" },
  { step: "yellow-150", fill: "warning-surface" },
  { step: "yellow-200", fill: "warning-border", line: "warning-border" },
  // As text, 300 and 400 are light greens on dark or green fills; see the
  // flagged rows.
  { step: "green-300", fill: "success", line: "success" },
  { step: "green-400", fill: "success", line: "success" },
  everywhere("green-500", "success"),
  everywhere("green-600", "success"),
  { step: "green-50", fill: "success-surface" },
  { step: "green-700", text: "success-foreground" },
  everywhere("emerald-500", "success"),
  everywhere("emerald-600", "success"),
  { step: "emerald-700", text: "success-foreground" },
  { step: "emerald-50", fill: "success-surface" },
  everywhere("blue-500", "info"),
  { step: "blue-600", text: "info-foreground" },
  { step: "blue-700", text: "info-foreground" },
  { step: "blue-800", text: "info-foreground" },
  { step: "blue-50", fill: "info-surface" },
  { step: "blue-100", fill: "info-surface" },
  { step: "blue-200", fill: "info/30", line: "info/30" },
  { step: "blue-300", fill: "info/50", line: "info/50" },
  everywhere("purple-300", "brand"),
  everywhere("purple-400", "brand"),
  everywhere("purple-500", "brand"),
  everywhere("purple-600", "brand-hover"),
  everywhere("purple-700", "accent-strong"),
  everywhere("purple-800", "accent-strong"),
  { step: "purple-50", fill: "accent-surface" },
  { step: "purple-100", fill: "accent-surface" },
  { step: "purple-200", fill: "brand/30", line: "brand/30" },
  everywhere("violet-500", "brand"),
  everywhere("violet-600", "brand"),
  everywhere("violet-700", "accent-strong"),
  { step: "violet-50", fill: "accent-surface" },
  { step: "violet-100", fill: "accent-surface" },
  { step: "violet-200", fill: "brand/20", line: "brand/20" },
];

/**
 * Plan 5.2 rows flagged as noticeably different: a new hue, a much paler
 * fill, a lighter label on a pale chip. `--flagged` only, once approved.
 * indigo-* has no single target (info or brand by use) and is left out.
 */
const FLAGGED_SNAPS: ColorSnap[] = [
  { step: "red-900", text: "error-foreground" },
  { step: "green-300", text: "success" },
  { step: "green-400", text: "success" },
  { step: "yellow-700", text: "warning-foreground" },
  everywhere("orange-500", "warning"),
  { step: "orange-600", text: "warning-foreground" },
  { step: "orange-200", fill: "warning-border", line: "warning-border" },
  { step: "green-100", fill: "success-surface", line: "success/40" },
  { step: "green-200", fill: "success-surface", line: "success/40" },
  { step: "emerald-300", text: "success/60" },
  everywhere("sky-500", "info"),
  { step: "sky-50", fill: "info-surface" },
  everywhere("violet-400", "brand"),
  { step: "violet-300", text: "brand/40", fill: "brand/40", line: "brand/40" },
  everywhere("rose-500", "error"),
  everywhere("rose-600", "error-hover"),
  { step: "rose-700", text: "error-foreground" },
  { step: "rose-50", fill: "error-surface" },
];

/** Plan 5.6: arbitrary letter-spacing and line-height to the nearest step. */
const TYPE_SNAPS: TokenGroup = {
  group: "Type (snapped)",
  kind: "typography",
  mappings: [
    { token: "tracking-caps", replaces: ["tracking-[0.14em]"] },
    {
      token: "tracking-eyebrow",
      replaces: [
        "tracking-[1.1px]",
        "tracking-[0.04em]",
        "tracking-[0.06785rem]",
      ],
    },
    {
      token: "tracking-display",
      replaces: [
        "tracking-[-0.055em]",
        "tracking-[-0.045em]",
        "tracking-[-0.035em]",
      ],
    },
    { token: "tracking-heading", replaces: ["tracking-[-0.6px]"] },
    {
      token: "tracking-title",
      replaces: [
        "tracking-[-0.08px]",
        "tracking-[-0.01313rem]",
        "tracking-[-0.033rem]",
        "tracking-[-0.02rem]",
      ],
    },
    { token: "leading-none", replaces: ["leading-[1]", "leading-[1.05]"] },
    { token: "leading-tight", replaces: ["leading-[1.2]"] },
    { token: "leading-snug", replaces: ["leading-[1.35]", "leading-[1.4]"] },
    { token: "leading-6.5", replaces: ["leading-[1.6rem]"] },
    { token: "leading-12", replaces: ["leading-[50px]"] },
  ],
};

export const SNAP_GROUPS: TokenGroup[] = [
  snapGroup("Greys (snapped)", GREY_SNAPS),
  snapGroup("Hues (snapped)", HUE_SNAPS),
  TYPE_SNAPS,
];

export const FLAGGED_SNAP_GROUPS: TokenGroup[] = [
  snapGroup("Hues (flagged)", FLAGGED_SNAPS),
];

/** Plan 5.6: raw sizes to the nearest size role, at most 4px away. */
export const SNAP_SIZE_TOKENS: TokenMapping[] = [
  { token: "text-tiny", replaces: ["text-[9px]"] },
  { token: "text-caption-sm", replaces: ["text-[0.7rem]", "text-[0.6785rem]"] },
  {
    token: "text-body-sm",
    replaces: ["text-[0.8rem]", "text-[12.5px]"],
  },
  { token: "text-body-lg", replaces: ["text-[17px]", "text-[1.125rem]"] },
  { token: "text-lead", replaces: ["text-[1.3rem]"] },
  { token: "text-heading-sm", replaces: ["text-[23px]"] },
  { token: "text-heading", replaces: ["text-[26px]", "text-[1.625rem]"] },
  { token: "text-title", replaces: ["text-[35px]", "text-[36px]"] },
  { token: "text-display", replaces: ["text-[3rem]"] },
];

export interface MapOptions {
  /** Add the snapped rows (nearest role) to the exact ones. */
  snap?: boolean;
  /** Add the flagged snapped rows too. */
  flagged?: boolean;
}

/** old class -> token class. Throws if a class is claimed twice. */
export function buildReplacementMap({
  snap = false,
  flagged = false,
}: MapOptions = {}) {
  const map = new Map<string, string>();
  const claim = (oldClass: string, token: string) => {
    const existing = map.get(oldClass);
    if (existing && existing !== token) {
      throw new Error(`${oldClass} maps to both ${existing} and ${token}`);
    }
    map.set(oldClass, token);
  };
  const groups = [
    ...TOKEN_GROUPS,
    ...(snap ? SNAP_GROUPS : []),
    ...(flagged ? FLAGGED_SNAP_GROUPS : []),
  ];
  for (const group of groups) {
    for (const mapping of group.mappings) {
      for (const oldClass of mapping.replaces) claim(oldClass, mapping.token);
    }
  }
  return map;
}

export function buildSizeMap({ snap = false }: MapOptions = {}) {
  const map = new Map<string, string>();
  const mappings = [...SIZE_TOKENS, ...(snap ? SNAP_SIZE_TOKENS : [])];
  for (const mapping of mappings) {
    for (const oldClass of mapping.replaces) map.set(oldClass, mapping.token);
  }
  return map;
}

const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx", ".mdx"]);
const SKIP_DIRS = new Set(["node_modules", ".next", "__generated__"]);

// A class list is a run of text between quotes or braces; its classes are
// the whitespace-separated words in it. Each word is parsed in code
// (variants, optional `!`, the utility, optional /opacity) rather than by one
// large regex, which keeps matching linear on any input.
const CLASS_LIST = /[^"'`{}]+/g;
const OPACITY = /^\/(?:[0-9.]+|\[[^\]]+\])$/;

/** Index of the last `ch` outside square brackets, or -1. */
function lastTopLevel(text: string, ch: string) {
  let depth = 0;
  let found = -1;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "[") depth++;
    else if (text[i] === "]") depth = Math.max(0, depth - 1);
    else if (text[i] === ch && depth === 0) found = i;
  }
  return found;
}

function parseClass(candidate: string) {
  const colon = lastTopLevel(candidate, ":");
  let variants = candidate.slice(0, colon + 1);
  let rest = candidate.slice(colon + 1);
  const bang = rest.startsWith("!") ? "!" : "";
  rest = rest.slice(bang.length);
  let opacity = "";
  const slash = lastTopLevel(rest, "/");
  if (slash > 0 && OPACITY.test(rest.slice(slash))) {
    opacity = rest.slice(slash);
    rest = rest.slice(0, slash);
  }
  // `placeholder:` mappings are keyed with their variant.
  let keyed = "";
  if (variants === "placeholder:" || variants.endsWith(":placeholder:")) {
    variants = variants.slice(0, -"placeholder:".length);
    keyed = "placeholder:";
  }
  return { variants, keyed, bang, utility: rest, opacity };
}

function isLeading(utility: string) {
  return utility.startsWith("leading-");
}

export interface Change {
  from: string;
  to: string;
  skipped?: string;
}

/** Rewrites one whitespace-separated class list. */
export function rewriteClassList(
  list: string,
  map: Map<string, string>,
  sizeMap: Map<string, string>,
) {
  const changes: Change[] = [];
  const words = list.split(/(\s+)/);
  const parsed = words.map((word) =>
    /^\s*$/.test(word) ? null : parseClass(word),
  );
  const leadings = parsed
    .map((p, i) => (p && isLeading(p.utility) ? { ...p, index: i } : null))
    .filter((p) => p !== null);

  const out = words.map((word, i) => {
    const p = parsed[i];
    if (!p) return word;
    const simple = map.get(`${p.keyed}${p.utility}`);
    if (simple && p.opacity && simple.includes("/")) {
      changes.push({
        from: word,
        to: word,
        skipped: `${simple} has its own opacity`,
      });
      return word;
    }
    if (simple) {
      const next = `${p.variants}${p.keyed}${p.bang}${simple.slice(p.keyed.length)}${p.opacity}`;
      changes.push({ from: word, to: next });
      return next;
    }
    const size = p.keyed || p.opacity ? undefined : sizeMap.get(p.utility);
    if (!size) return word;
    const sameScope = leadings.filter((l) => l.variants === p.variants);
    const otherScope = leadings.filter((l) => l.variants !== p.variants);
    const skip = (reason: string) => {
      changes.push({ from: word, to: word, skipped: reason });
      return word;
    };
    if (p.bang && leadings.length > 0) {
      return skip("!important size next to a leading-*");
    }
    if (sameScope.some((l) => l.index < i)) {
      return skip("a leading-* comes before the size");
    }
    if (p.variants && otherScope.length > 0) {
      return skip("a leading-* in another variant scope");
    }
    const swapped = `${p.variants}${p.bang}${size}`;
    const next = sameScope.length
      ? swapped
      : `${swapped} ${p.variants}${p.bang}leading-inherit`;
    changes.push({ from: word, to: next });
    return next;
  });
  return { output: out.join(""), changes };
}

export function rewriteSource(
  source: string,
  map: Map<string, string>,
  sizeMap: Map<string, string>,
) {
  const changes: (Change & { line: number })[] = [];
  const output = source.replace(CLASS_LIST, (list: string, offset: number) => {
    const result = rewriteClassList(list, map, sizeMap);
    if (result.changes.length === 0) return list;
    const line = source.slice(0, offset).split("\n").length;
    for (const change of result.changes) changes.push({ ...change, line });
    return result.output;
  });
  return { output, changes };
}

function collectFiles(path: string, out: string[]) {
  const stat = statSync(path);
  if (stat.isDirectory()) {
    for (const entry of readdirSync(path)) {
      if (SKIP_DIRS.has(entry)) continue;
      collectFiles(join(path, entry), out);
    }
    return;
  }
  if (SOURCE_EXTENSIONS.has(extname(path))) out.push(path);
}

function main(argv: string[]) {
  const write = argv.includes("--write");
  const flagged = argv.includes("--flagged");
  const targets = argv.filter((arg) => !arg.startsWith("--"));
  if (targets.length === 0) {
    console.error(
      "usage: npx tsx scripts/codemod-tokens.ts [--write] [--exact | --flagged] <file|dir>...",
    );
    return 2;
  }
  const options = { snap: !argv.includes("--exact"), flagged };
  const map = buildReplacementMap(options);
  const sizeMap = buildSizeMap(options);
  const files: string[] = [];
  for (const target of targets) collectFiles(target, files);

  let total = 0;
  let skipped = 0;
  for (const file of files) {
    const source = readFileSync(file, "utf8");
    const { output, changes } = rewriteSource(source, map, sizeMap);
    if (changes.length === 0) continue;
    for (const change of changes) {
      const where = `${relative(process.cwd(), file)}:${change.line}`;
      if (change.skipped) {
        skipped++;
        console.log(`${where}  ${change.from} SKIPPED: ${change.skipped}`);
      } else {
        total++;
        console.log(`${where}  ${change.from} -> ${change.to}`);
      }
    }
    if (write) writeFileSync(file, output);
  }
  console.log(
    `${total} replacement(s), ${skipped} class(es) skipped, in ${files.length} file(s)${write ? "" : " (dry run, pass --write to apply)"}`,
  );
  return 0;
}

if (process.argv[1]?.endsWith("codemod-tokens.ts")) {
  process.exitCode = main(process.argv.slice(2));
}
