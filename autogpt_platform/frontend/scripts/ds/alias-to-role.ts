/**
 * Renames the shadcn colour names (`bg-background`, `text-muted-foreground`,
 * `border-border`, `ring-ring`, ...) to the role each one aliased, so the app
 * names one colour system (SECRT-3154). Each alias already rendered its role's
 * value, so the rename moves no pixel; scripts/check-token-equivalence.ts
 * proves it against a base checkout.
 *
 *   npx tsx scripts/ds/alias-to-role.ts <file|dir>...            # dry run
 *   npx tsx scripts/ds/alias-to-role.ts --write <file|dir>...    # rewrite
 *
 * Variants (`hover:`, `[&_svg]:`), `!important` and opacity (`/50`) are kept.
 * `var(--color-<alias>)` in inline styles and arbitrary values is renamed the
 * same way.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import { rewriteSource, type Change } from "../codemod-tokens.ts";
import { COLOUR_UTILITIES, expandNames } from "./colour-utilities.ts";

/** The 27 shadcn names and the role each aliased. */
export const ALIAS_ROLES: Record<string, string> = {
  background: "canvas",
  foreground: "foreground-base",
  card: "surface",
  "card-foreground": "foreground-base",
  popover: "surface",
  "popover-foreground": "foreground-base",
  primary: "action",
  "primary-foreground": "action-foreground",
  secondary: "surface-hover",
  "secondary-foreground": "foreground-strong",
  muted: "surface-hover",
  "muted-foreground": "foreground-tertiary",
  accent: "brand",
  "accent-foreground": "action-foreground",
  destructive: "error",
  "destructive-foreground": "action-foreground",
  border: "border-base",
  input: "border-strong",
  ring: "focus",
  sidebar: "surface-muted",
  "sidebar-foreground": "foreground-emphasis",
  "sidebar-primary": "action",
  "sidebar-primary-foreground": "action-foreground",
  "sidebar-accent": "surface-hover",
  "sidebar-accent-foreground": "foreground-strong",
  "sidebar-border": "border-base",
  "sidebar-ring": "info",
};

export const ALIAS_NAMES = Object.keys(ALIAS_ROLES);

export function buildAliasMap() {
  // `accent-` (accent-color) is left out: no app class sets a shadcn name
  // with it, and the word turns up in prose ("accent-foreground").
  const map = expandNames(
    ALIAS_ROLES,
    COLOUR_UTILITIES.filter((utility) => utility !== "accent"),
  );
  // rewriteSource keys a `placeholder:` class with its variant.
  for (const [name, role] of Object.entries(ALIAS_ROLES))
    map.set(`placeholder:text-${name}`, `placeholder:text-${role}`);
  return map;
}

const ALIAS_VARIABLE = new RegExp(
  String.raw`var\(--color-(${[...ALIAS_NAMES]
    .sort((a, b) => b.length - a.length)
    .join("|")})\)`,
  "g",
);

export function rewriteAliases(source: string, map = buildAliasMap()) {
  const classes = rewriteSource(source, map, new Map());
  const changes: (Change & { line: number })[] = [...classes.changes];
  const output = classes.output.replace(
    ALIAS_VARIABLE,
    (match: string, name: string, offset: number) => {
      const to = `var(--color-${ALIAS_ROLES[name]})`;
      const line = classes.output.slice(0, offset).split("\n").length;
      changes.push({ from: match, to, line });
      return to;
    },
  );
  return { output, changes };
}

const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".mdx", ".css"]);
const SKIP_DIRS = new Set(["node_modules", ".next", "__generated__"]);
// The token file defines the aliases streamdown reads; never rewrite it.
const SKIP_FILES = new Set(["theme.css"]);

export function collectFiles(path: string, out: string[] = []) {
  if (statSync(path).isDirectory()) {
    for (const entry of readdirSync(path)) {
      if (SKIP_DIRS.has(entry)) continue;
      collectFiles(join(path, entry), out);
    }
  } else if (
    SOURCE_EXTENSIONS.has(extname(path)) &&
    !SKIP_FILES.has(path.split("/").pop() ?? "")
  ) {
    out.push(path);
  }
  return out;
}

function main(argv: string[]) {
  const write = argv.includes("--write");
  const targets = argv.filter((arg) => !arg.startsWith("--"));
  if (targets.length === 0) {
    console.error(
      "usage: npx tsx scripts/ds/alias-to-role.ts [--write] <file|dir>...",
    );
    return 2;
  }
  const map = buildAliasMap();
  let total = 0;
  let touched = 0;
  for (const file of targets.flatMap((target) => collectFiles(target))) {
    const source = readFileSync(file, "utf8");
    const { output, changes } = rewriteAliases(source, map);
    if (changes.length === 0) continue;
    touched++;
    for (const change of changes) {
      total++;
      console.log(
        `${relative(process.cwd(), file)}:${change.line}  ${change.from} -> ${change.to}`,
      );
    }
    if (write) writeFileSync(file, output);
  }
  console.log(
    `${total} rename(s) in ${touched} file(s)${write ? "" : " (dry run, pass --write to apply)"}`,
  );
  return 0;
}

if (process.argv[1]?.endsWith("alias-to-role.ts")) {
  process.exitCode = main(process.argv.slice(2));
}
