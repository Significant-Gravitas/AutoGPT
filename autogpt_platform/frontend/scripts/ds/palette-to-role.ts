/**
 * Swaps Tailwind palette classes (`text-orange-600`, `bg-gray-100`,
 * `shadow-black/10`) for colour roles, per the reviewed table in
 * palette-map.json (SECRT-3154). Files and classes in ds-colour-exempt.json
 * are left alone.
 *
 *   npx tsx scripts/ds/palette-to-role.ts <file|dir>...            # dry run
 *   npx tsx scripts/ds/palette-to-role.ts --write <file|dir>...    # rewrite
 *   npx tsx scripts/ds/palette-to-role.ts --unmapped <file|dir>... # list what no row covers
 *
 * Variants, `!important` and opacity modifiers are kept. A row whose role
 * carries its own opacity (`orange/10`) is skipped, and reported, on a class
 * that already has one (`bg-orange-50/60`): rewrite those by hand.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { rewriteSource } from "../codemod-tokens.ts";
import { collectFiles } from "./alias-to-role.ts";
import { isExempt, isWholeFileExempt } from "./colour-exempt.ts";
import {
  colourClassPattern,
  FILL_UTILITIES,
  LINE_UTILITIES,
  TEXT_UTILITIES,
} from "./colour-utilities.ts";

interface Row {
  text?: string;
  fill?: string;
  line?: string;
  change: "none" | "slight" | "bigger";
  note?: string;
}

interface PaletteMap {
  classes: Record<string, string>;
  steps: Record<string, Row>;
}

export const PALETTE_MAP = JSON.parse(
  readFileSync(new URL("./palette-map.json", import.meta.url), "utf8"),
) as PaletteMap;

/** Every palette name a class can use: the preset's families and keywords. */
export const PALETTE_VALUES = String.raw`(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|customGray)-\d{2,3}|white|black|textGrey|textBlack|bgLightGrey`;

/** old class -> role class, from the table. */
export function buildPaletteMap(table: PaletteMap = PALETTE_MAP) {
  const map = new Map<string, string>();
  const add = (utilities: string[], step: string, role?: string) => {
    if (!role) return;
    for (const utility of utilities)
      map.set(`${utility}-${step}`, `${utility}-${role}`);
  };
  for (const [step, row] of Object.entries(table.steps)) {
    add(TEXT_UTILITIES, step, row.text);
    add(FILL_UTILITIES, step, row.fill);
    add(LINE_UTILITIES, step, row.line);
    if (row.text)
      map.set(`placeholder:text-${step}`, `placeholder:text-${row.text}`);
  }
  for (const [from, to] of Object.entries(table.classes)) map.set(from, to);
  return map;
}

/** The palette classes in `source`, as `<utility>-<value>`. */
export function paletteClasses(source: string) {
  return [...source.matchAll(colourClassPattern(PALETTE_VALUES))].map(
    (match) => `${match[2]}-${match[3]}`,
  );
}

/** The map without the classes `file` may keep. */
function mapFor(file: string, map: Map<string, string>) {
  const kept = new Map(map);
  for (const key of map.keys()) {
    const bare = key.replace(/^placeholder:/, "");
    if (isExempt(file, bare)) kept.delete(key);
  }
  return kept;
}

function main(argv: string[]) {
  const write = argv.includes("--write");
  const unmapped = argv.includes("--unmapped");
  const targets = argv.filter((arg) => !arg.startsWith("--"));
  if (targets.length === 0) {
    console.error(
      "usage: npx tsx scripts/ds/palette-to-role.ts [--write | --unmapped] <file|dir>...",
    );
    return 2;
  }
  const map = buildPaletteMap();
  let total = 0;
  let skipped = 0;
  const left = new Map<string, string[]>();
  for (const file of targets.flatMap((target) => collectFiles(target))) {
    const name = relative(process.cwd(), file);
    if (isWholeFileExempt(name)) continue;
    const source = readFileSync(file, "utf8");
    const { output, changes } = rewriteSource(
      source,
      mapFor(name, map),
      new Map(),
    );
    for (const change of changes) {
      if (change.skipped) skipped++;
      else total++;
      if (!unmapped)
        console.log(
          `${name}:${change.line}  ${change.from} ${change.skipped ? `SKIPPED: ${change.skipped}` : `-> ${change.to}`}`,
        );
    }
    for (const found of paletteClasses(output)) {
      if (isExempt(name, found)) continue;
      left.set(found, [...(left.get(found) ?? []), name]);
    }
    if (write && changes.length > 0) writeFileSync(file, output);
  }
  if (unmapped)
    for (const [found, files] of [...left].sort())
      console.log(`${found}  ${[...new Set(files)].join(" ")}`);
  console.log(
    `${total} replacement(s), ${skipped} skipped, ${left.size} palette class(es) no row covers${write ? "" : " (dry run, pass --write to apply)"}`,
  );
  return 0;
}

if (process.argv[1]?.endsWith("palette-to-role.ts")) {
  process.exitCode = main(process.argv.slice(2));
}
