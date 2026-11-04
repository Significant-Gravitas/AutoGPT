/**
 * Reads ds-colour-exempt.json: the files, or the classes and hex values
 * within a file, that may keep a raw palette class or a hex colour. Read by
 * design-system/no-palette-colours and no-hex-colours, by the palette codemod
 * (which leaves them alone) and by scripts/__tests__/colour-exempt.test.ts.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export interface ExemptEntry {
  /** A glob relative to the frontend folder (`*`, `**`, `?`, `{a,b}`). */
  files: string;
  /**
   * Only these classes (`text-emerald-300`, no variant or opacity) and hex
   * values (`#ffffff`, any case) are allowed. Omitted: the whole file is.
   */
  allow?: string[];
  reason: string;
}

export const EXEMPT_FILE = fileURLToPath(
  new URL("../../ds-colour-exempt.json", import.meta.url),
);

let entries: (ExemptEntry & { pattern: RegExp })[] | null = null;

export function globToRegExp(glob: string) {
  let source = "";
  for (let i = 0; i < glob.length; i++) {
    const ch = glob[i];
    if (ch === "*" && glob[i + 1] === "*") {
      // `**/` matches any number of folders, including none.
      source += glob[i + 2] === "/" ? "(?:.*/)?" : ".*";
      i += glob[i + 2] === "/" ? 2 : 1;
    } else if (ch === "*") source += "[^/]*";
    else if (ch === "?") source += "[^/]";
    else if (ch === "{") {
      const end = glob.indexOf("}", i);
      const options = glob.slice(i + 1, end).split(",");
      source += `(?:${options.map(escape).join("|")})`;
      i = end;
    } else source += escape(ch);
  }
  return new RegExp(`^${source}$`);
}

function escape(text: string) {
  return text.replace(/[.+^${}()|[\]\\]/g, "\\$&");
}

export function exemptEntries() {
  entries ??= (
    JSON.parse(readFileSync(EXEMPT_FILE, "utf8")) as { exempt: ExemptEntry[] }
  ).exempt.map((entry) => ({ ...entry, pattern: globToRegExp(entry.files) }));
  return entries;
}

/**
 * Whether `value` (a palette class without variants or opacity, or a hex
 * colour) may stay in `file`, a path relative to the frontend folder.
 */
export function isExempt(file: string, value: string) {
  const wanted = value.toLowerCase();
  return exemptEntries().some(
    (entry) =>
      entry.pattern.test(file) &&
      (!entry.allow ||
        entry.allow.some((allowed) => allowed.toLowerCase() === wanted)),
  );
}

/** Whether every colour in `file` is exempt. */
export function isWholeFileExempt(file: string) {
  return exemptEntries().some(
    (entry) => !entry.allow && entry.pattern.test(file),
  );
}
