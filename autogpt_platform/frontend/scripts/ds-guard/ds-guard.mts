/**
 * Design-system ratchet (SECRT-3097, wave 0).
 *
 *   node scripts/ds-guard/ds-guard.mts            check against baseline.json
 *   node scripts/ds-guard/ds-guard.mts --update   rewrite baseline.json
 *
 * Counts design-system violations per file under src/ and compares them with
 * the committed baseline.json. Existing violations never fail: the check fails
 * only when a file has more of one kind than its baseline allows, or when a
 * file missing from the baseline has any. Lower counts pass with a hint to run
 * --update so they cannot grow back.
 *
 * Node strips the types when it runs this file, so it stays one self-contained
 * module with no dependencies, no relative imports and no TypeScript-only
 * syntax. CI runs it without installing node_modules.
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type RuleId =
  | "palette-color"
  | "arbitrary-color"
  | "dark-variant"
  | "legacy-import"
  | "legacy-button-variant";

export type Counts = Partial<Record<RuleId, number>>;
export type Baseline = Record<string, Counts>;

export const FRONTEND_ROOT = fileURLToPath(new URL("../..", import.meta.url));
export const BASELINE_FILE = fileURLToPath(
  new URL("./baseline.json", import.meta.url),
);

export const RULES: Record<RuleId, string> = {
  "palette-color":
    "Raw Tailwind palette class (bg-zinc-200, text-neutral-500, border-red-300...). Use a colour role token (bg-surface, text-foreground-secondary, border-border-base...).",
  "arbitrary-color":
    "Arbitrary colour value (bg-[#fff], text-[rgb(...)], shadow-[..._#000]...). Use a colour role token, or add one to globals.css + tailwind.config.ts.",
  "dark-variant":
    "`dark:` class. Dark mode comes from the colour role tokens, not per-class overrides.",
  "legacy-import":
    "Import from a legacy path (__legacy__, legacy/, LegacyButton...). Use the design-system atoms, molecules and organisms in src/components.",
  "legacy-button-variant":
    'Legacy Button variant or size ("legacy-default", "legacy-outline", "legacy-sm"...), which renders LegacyButton. Use a design-system Button variant.',
};

const RULE_IDS = Object.keys(RULES) as RuleId[];

const PALETTES = [
  "slate",
  "gray",
  "zinc",
  "neutral",
  "stone",
  "red",
  "orange",
  "amber",
  "yellow",
  "lime",
  "green",
  "emerald",
  "teal",
  "cyan",
  "sky",
  "blue",
  "indigo",
  "violet",
  "purple",
  "fuchsia",
  "pink",
  "rose",
].join("|");

// Every Tailwind utility that takes a colour.
const COLOR_UTILITIES = [
  "bg",
  "text",
  "border(?:-[xytrblse])?",
  "ring(?:-offset)?",
  "inset-ring",
  "outline",
  "divide",
  "fill",
  "stroke",
  "from",
  "via",
  "to",
  "decoration",
  "placeholder",
  "caret",
  "accent",
  "shadow",
  "inset-shadow",
  "drop-shadow",
].join("|");

// A class starts after whitespace, a quote, a variant's `:` or an `!`, never
// inside a longer name (`foo-bg-red-500`, `myText-red-500`).
const START = String.raw`(?<![\w-])`;
const END = String.raw`(?![\w-])`;

const PALETTE_CLASS = new RegExp(
  `${START}(?:${COLOR_UTILITIES})-(?:${PALETTES})-(?:50|[1-9]00|950)${END}`,
  "g",
);
const ARBITRARY_VALUE = new RegExp(
  String.raw`${START}(?:${COLOR_UTILITIES})-\[([^\]\s]+)\]`,
  "g",
);
const COLOR_VALUE =
  /#[0-9a-f]{3,8}(?![0-9a-z])|(?<![a-z-])(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color-mix|color)\(|^color:/i;
const DARK_VARIANT = new RegExp(String.raw`${START}dark:(?=[\w[!*@(-])`, "g");
const MODULE_SPECIFIER =
  /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*)(["'`])([^"'`\n]+)\1/g;
const LEGACY_SEGMENT = /^(?:__legacy__|legacy|Legacy[A-Z]\w*)$/;
const LEGACY_BUTTON_VARIANT =
  /(["'`])legacy-(?:default|destructive|outline|ghost|link|sm|icon)\1/g;

function countMatches(source: string, pattern: RegExp) {
  return source.match(pattern)?.length ?? 0;
}

function isLegacySpecifier(specifier: string) {
  return specifier
    .replace(/\.[cm]?[jt]sx?$/, "")
    .split("/")
    .some((segment) => LEGACY_SEGMENT.test(segment));
}

export function countViolations(source: string): Counts {
  let arbitrary = 0;
  for (const match of source.matchAll(ARBITRARY_VALUE)) {
    if (COLOR_VALUE.test(match[1])) arbitrary++;
  }
  let legacyImports = 0;
  for (const match of source.matchAll(MODULE_SPECIFIER)) {
    if (isLegacySpecifier(match[2])) legacyImports++;
  }
  const counts: Counts = {
    "palette-color": countMatches(source, PALETTE_CLASS),
    "arbitrary-color": arbitrary,
    "dark-variant": countMatches(source, DARK_VARIANT),
    "legacy-import": legacyImports,
    "legacy-button-variant": countMatches(source, LEGACY_BUTTON_VARIANT),
  };
  for (const rule of RULE_IDS) {
    if (!counts[rule]) delete counts[rule];
  }
  return counts;
}

const SCANNED = /\.(?:[cm]?[jt]sx?|css|mdx)$/;

// Tests and test helpers assert on class names and mock legacy modules; they
// render nothing users see.
export function isScanned(file: string) {
  return (
    SCANNED.test(file) &&
    !file.endsWith(".d.ts") &&
    !/\.(?:test|spec)\.[cm]?[jt]sx?$/.test(file) &&
    !file.split("/").includes("__tests__") &&
    !file.startsWith("src/app/api/__generated__/") &&
    !file.startsWith("src/tests/") &&
    !file.startsWith("src/playwright/")
  );
}

export function scan(root = FRONTEND_ROOT): Baseline {
  const result: Baseline = {};
  const files = readdirSync(path.join(root, "src"), { recursive: true })
    .map((file) => `src/${String(file).split(path.sep).join("/")}`)
    .filter(isScanned)
    .sort();
  for (const file of files) {
    const counts = countViolations(readFileSync(path.join(root, file), "utf8"));
    if (Object.keys(counts).length > 0) result[file] = counts;
  }
  return result;
}

export interface Change {
  file: string;
  rule: RuleId;
  allowed: number;
  found: number;
}

export function compare(baseline: Baseline, current: Baseline) {
  const regressions: Change[] = [];
  const improvements: Change[] = [];
  const files = new Set([...Object.keys(baseline), ...Object.keys(current)]);
  for (const file of [...files].sort()) {
    for (const rule of RULE_IDS) {
      const allowed = baseline[file]?.[rule] ?? 0;
      const found = current[file]?.[rule] ?? 0;
      if (found > allowed) regressions.push({ file, rule, allowed, found });
      if (found < allowed) improvements.push({ file, rule, allowed, found });
    }
  }
  return { regressions, improvements };
}

export function serialize(baseline: Baseline) {
  return `${JSON.stringify(baseline, null, 2)}\n`;
}

function totals(baseline: Baseline) {
  return RULE_IDS.map((rule) => {
    const total = Object.values(baseline).reduce(
      (sum, counts) => sum + (counts[rule] ?? 0),
      0,
    );
    return `  ${rule}: ${total}`;
  }).join("\n");
}

function formatChanges(changes: Change[]) {
  return changes
    .map(({ file, rule, allowed, found }) => {
      const verb = allowed === 0 ? "new" : `was ${allowed}`;
      return `  ${file}: ${found} ${rule} (${verb})`;
    })
    .join("\n");
}

function main(args: string[]) {
  const current = scan();
  if (args.includes("--update")) {
    writeFileSync(BASELINE_FILE, serialize(current));
    console.log(
      `ds-guard: wrote baseline.json for ${Object.keys(current).length} files.\n${totals(current)}`,
    );
    return 0;
  }

  const baseline: Baseline = JSON.parse(readFileSync(BASELINE_FILE, "utf8"));
  const { regressions, improvements } = compare(baseline, current);

  if (improvements.length > 0) {
    console.log(
      `ds-guard: ${improvements.length} count(s) went down. Lock them in with \`pnpm ds:guard:update\` and commit scripts/ds-guard/baseline.json so they cannot grow back:\n${formatChanges(improvements)}\n`,
    );
  }

  if (regressions.length === 0) {
    console.log(`ds-guard: no new violations.\n${totals(current)}`);
    return 0;
  }

  if (process.env.GITHUB_ACTIONS) {
    for (const { file, rule, allowed, found } of regressions) {
      console.log(
        `::error file=autogpt_platform/frontend/${file},title=ds-guard ${rule}::${found} ${rule} where baseline.json allows ${allowed}. ${RULES[rule]}`,
      );
    }
  }
  const rules = [...new Set(regressions.map((change) => change.rule))];
  console.error(
    [
      `ds-guard: ${regressions.length} new design-system violation(s):`,
      formatChanges(regressions),
      "",
      ...rules.map((rule) => `  ${rule}: ${RULES[rule]}`),
      "",
      "Fix the new violations. If a count only moved (a renamed or split file), run `pnpm ds:guard:update` and commit scripts/ds-guard/baseline.json; design-system owners review that file.",
    ].join("\n"),
  );
  return 1;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
