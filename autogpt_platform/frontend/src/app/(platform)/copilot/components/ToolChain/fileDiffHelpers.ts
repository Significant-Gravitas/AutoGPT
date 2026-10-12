export interface DiffRow {
  old: number | null;
  cur: number | null;
  type: "ctx" | "add" | "del";
  text: string;
}

export interface ParsedDiff {
  rows: DiffRow[];
  added: number;
  removed: number;
  truncated: boolean;
}

export const MAX_RENDERED_DIFF_ROWS = 500;

export function isDiffText(value: unknown): value is string {
  if (typeof value !== "string") return false;
  return /^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@/m.test(value);
}

export function parseUnifiedDiff(diff: string): ParsedDiff {
  const rows: DiffRow[] = [];
  let added = 0;
  let removed = 0;
  let totalRows = 0;
  let inHunk = false;
  let oldLn = 1;
  let newLn = 1;
  for (const line of diff.split("\n")) {
    const hunk = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunk) {
      inHunk = true;
      oldLn = Number(hunk[1]);
      newLn = Number(hunk[2]);
      continue;
    }
    if (
      !inHunk ||
      line.startsWith("--- ") ||
      line.startsWith("+++ ") ||
      line === "\\ No newline at end of file"
    ) {
      continue;
    }
    let row: DiffRow;
    if (line.startsWith("-")) {
      removed++;
      row = { old: oldLn++, cur: null, type: "del", text: line.slice(1) };
    } else if (line.startsWith("+")) {
      added++;
      row = { old: null, cur: newLn++, type: "add", text: line.slice(1) };
    } else {
      row = {
        old: oldLn++,
        cur: newLn++,
        type: "ctx",
        text: line.startsWith(" ") ? line.slice(1) : line,
      };
    }
    totalRows++;
    if (rows.length < MAX_RENDERED_DIFF_ROWS) rows.push(row);
  }
  return { rows, added, removed, truncated: rows.length < totalRows };
}
