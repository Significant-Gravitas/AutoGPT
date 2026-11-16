/**
 * The type scale and the reviewed mapping from Tailwind's sizes onto it
 * (SECRT-3152). The scale's values live in src/app/tokens/typography.css;
 * this file names its steps for the codemod below, the lint rule
 * (design-system/no-raw-text-size) and scripts/__tests__/type-scale.test.ts.
 *
 *   npx tsx scripts/ds/text-size-scale.ts <file|dir>...           # dry run
 *   npx tsx scripts/ds/text-size-scale.ts --write <file|dir>...   # rewrite
 *
 * The codemod rewrites class names inside string literals and template text
 * only (comments are left alone), keeping variant prefixes (`md:`, `hover:`,
 * `[&_svg]:`) and `!important`. Each Tailwind size moves to the step with the
 * same or nearest size; the step's own line height comes with it, so text
 * that used Tailwind's line height moves by up to 4px per line, while text
 * with an explicit `leading-*` keeps it.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

export interface TypeStep {
  name: string;
  size: string;
  lineHeight: string;
}

/** Smallest to largest, as in tokens/typography.css. */
export const TYPE_SCALE: TypeStep[] = [
  { name: "tiny", size: "0.625rem", lineHeight: "0.875rem" },
  { name: "caption-sm", size: "0.6875rem", lineHeight: "1rem" },
  { name: "caption", size: "0.75rem", lineHeight: "1.125rem" },
  { name: "body-sm", size: "0.8125rem", lineHeight: "1.25rem" },
  { name: "body", size: "0.875rem", lineHeight: "1.375rem" },
  { name: "body-md", size: "0.9375rem", lineHeight: "1.5rem" },
  { name: "body-lg", size: "1rem", lineHeight: "1.625rem" },
  { name: "lead", size: "1.25rem", lineHeight: "1.75rem" },
  { name: "heading-sm", size: "1.375rem", lineHeight: "1.5rem" },
  { name: "heading-md", size: "1.5rem", lineHeight: "2rem" },
  { name: "heading", size: "1.75rem", lineHeight: "2.5rem" },
  { name: "title", size: "2rem", lineHeight: "2.5rem" },
  { name: "display", size: "2.75rem", lineHeight: "3.5rem" },
  { name: "hero", size: "4rem", lineHeight: "1" },
];

export interface RawSize {
  /** The scale class that replaces it. */
  token: string;
  /** Tailwind 3's size / line height, and what changes. */
  note: string;
}

/** Every Tailwind size, and the step of the scale it becomes. */
export const RAW_TEXT_SIZES: Record<string, RawSize> = {
  "text-xs": { token: "text-caption", note: "12/16 -> 12/18, line +2px" },
  "text-sm": { token: "text-body", note: "14/20 -> 14/22, line +2px" },
  "text-base": { token: "text-body-lg", note: "16/24 -> 16/26, line +2px" },
  "text-lg": { token: "text-lead", note: "18/28 -> 20/28, size +2px" },
  "text-xl": { token: "text-lead", note: "20/28, the same" },
  "text-2xl": { token: "text-heading-md", note: "24/32, the same" },
  "text-3xl": {
    token: "text-heading",
    note: "30/36 -> 28/40, size -2px, line +4px",
  },
  "text-4xl": { token: "text-title", note: "36/40 -> 32/40, size -4px" },
  "text-5xl": { token: "text-display", note: "48/48 -> 44/56, size -4px" },
  "text-6xl": { token: "text-hero", note: "60/60 -> 64/64, size +4px" },
  "text-7xl": { token: "text-hero", note: "72/72 -> 64/64, size -8px" },
  "text-8xl": { token: "text-hero", note: "96/96 -> 64/64, size -32px" },
  "text-9xl": { token: "text-hero", note: "128/128 -> 64/64, size -64px" },
};

const PX_PER_REM = 16;

function toPx(value: string) {
  const match = /^([0-9]*\.?[0-9]+)(px|rem|em)$/.exec(value);
  if (!match) return null;
  const amount = Number(match[1]);
  return match[2] === "px" ? amount : amount * PX_PER_REM;
}

/** The step whose size is nearest a pixel size. */
export function nearestStep(px: number) {
  let best = TYPE_SCALE[0];
  for (const step of TYPE_SCALE) {
    const distance = Math.abs((toPx(step.size) ?? 0) - px);
    if (distance < Math.abs((toPx(best.size) ?? 0) - px)) best = step;
  }
  return best;
}

// `text-sm`, `md:text-sm`, `!text-sm`, `[&_svg]:text-sm`; also the arbitrary
// lengths (`text-[13px]`, `text-[0.8rem]`, `text-[length:...]`).
const NAMED = Object.keys(RAW_TEXT_SIZES)
  .map((name) => name.slice("text-".length))
  .join("|");
export const RAW_TEXT_SIZE = new RegExp(
  `(?<![\\w-])((?:[^\\s"'\`{}]*:)?!?)text-(?:(${NAMED})|\\[(length:[^\\]\\s]+|[0-9]*\\.?[0-9]+(?:px|rem|em)|(?:clamp|calc|min|max)\\([^\\]\\s]*)\\])(?=$|[^\\w\\[-])`,
  "g",
);

/** What to write instead of a raw size, for the lint message. */
export function rawTextSizeAdvice(named: string | undefined, value?: string) {
  if (named) {
    const raw = RAW_TEXT_SIZES[`text-${named}`];
    return `Use "${raw.token}" (or a Text variant at that size) instead: the type scale's step for it (${raw.note}).`;
  }
  const px = value ? toPx(value) : null;
  if (px === null)
    return "Use a step of the type scale (src/app/tokens/typography.css) instead.";
  const step = nearestStep(px);
  return `Use "text-${step.name}" (or a Text variant at that size) instead: the type scale's nearest step (${step.size}).`;
}

/** Rewrites the raw named sizes in one string; arbitrary sizes are left. */
export function rewriteSizes(text: string) {
  const changes: string[] = [];
  const output = text.replace(
    RAW_TEXT_SIZE,
    (match, prefix: string, named: string | undefined) => {
      if (!named) return match;
      const to = `${prefix}${RAW_TEXT_SIZES[`text-${named}`].token}`;
      changes.push(`${match} -> ${to}`);
      return to;
    },
  );
  return { output, changes };
}

/** Rewrites every string literal and template text in a TS/TSX source. */
export function rewriteSource(source: string, fileName: string) {
  const file = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    fileName.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const edits: { start: number; end: number; text: string }[] = [];
  const changes: string[] = [];
  function visit(node: ts.Node) {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) ||
      ts.isTemplateTail(node)
    ) {
      // The raw text between the delimiters, so escapes stay as written.
      const start = node.getStart(file);
      const raw = source.slice(start, node.getEnd());
      const rewritten = rewriteSizes(raw);
      if (rewritten.changes.length > 0) {
        edits.push({ start, end: node.getEnd(), text: rewritten.output });
        changes.push(...rewritten.changes);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(file);
  let output = source;
  for (const edit of edits.sort((a, b) => b.start - a.start))
    output = output.slice(0, edit.start) + edit.text + output.slice(edit.end);
  return { output, changes };
}

const SOURCE_EXTENSIONS = new Set([".ts", ".tsx"]);
const SKIP_DIRS = new Set(["node_modules", ".next", "__generated__"]);

function* sourceFiles(path: string): Generator<string> {
  if (statSync(path).isFile()) {
    if (SOURCE_EXTENSIONS.has(extname(path))) yield path;
    return;
  }
  for (const entry of readdirSync(path)) {
    if (SKIP_DIRS.has(entry)) continue;
    yield* sourceFiles(join(path, entry));
  }
}

function main(args: string[]) {
  const write = args.includes("--write");
  const paths = args.filter((arg) => !arg.startsWith("--"));
  let total = 0;
  for (const path of paths) {
    for (const file of sourceFiles(path)) {
      const source = readFileSync(file, "utf8");
      const { output, changes } = rewriteSource(source, file);
      if (changes.length === 0) continue;
      total += changes.length;
      console.log(`${relative(process.cwd(), file)}: ${changes.join(", ")}`);
      if (write) writeFileSync(file, output);
    }
  }
  console.log(`${total} size class(es) ${write ? "rewritten" : "to rewrite"}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href)
  main(process.argv.slice(2));
