import { looksTechnical } from "./plainText";

export type DataSummary =
  | {
      kind: "table";
      rows: number;
      columns: string[];
      delimiter: "," | "\t";
      // The first rows, cell by cell, for the preview.
      sample: string[][];
    }
  | { kind: "json"; fields: string[]; sample: [string, string][] }
  | { kind: "json-list"; items: number };

const MAX_HEADER_CELL = 40;
const MIN_TABLE_LINES = 3;
const SAMPLE_ROWS = 3;
const SAMPLE_COLUMNS = 4;
const SAMPLE_FIELDS = 4;

/** The columns a preview shows, first few: those whose name and every
 *  sampled value are plain. A column of paths, commands or ids is left out
 *  whole; its values stay behind technical details and in the download. */
export function previewColumns(
  columns: string[],
  sample: string[][] = [],
): number[] {
  return columns
    .map((name, index) => ({ name, index }))
    .filter(
      ({ name, index }) =>
        !looksTechnical(name) &&
        sample.every((cells) => !looksTechnical(cells[index] ?? "")),
    )
    .slice(0, SAMPLE_COLUMNS)
    .map(({ index }) => index);
}

/** What a command printed, when it is a table (CSV/TSV) or JSON rather than
 *  prose: enough to show it as a chip instead of dumping it into the chat. */
export function summarizeData(text: string): DataSummary | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  return summarizeJSON(trimmed) ?? summarizeTable(trimmed);
}

export function describeData(summary: DataSummary): {
  title: string;
  detail: string;
} {
  if (summary.kind === "table") {
    return {
      title: "Table",
      detail: `${plural(summary.rows, "row")} · ${listPreview(summary.columns, "column")}`,
    };
  }
  if (summary.kind === "json") {
    return {
      title: "Data",
      detail:
        summary.fields.length > 0
          ? `${plural(summary.fields.length, "field")} · ${listPreview(summary.fields, "field")}`
          : "No fields",
    };
  }
  return { title: "Data", detail: plural(summary.items, "item") };
}

/** The file a preview's Download button saves: what the step printed. */
export function dataFile(summary: DataSummary): {
  name: string;
  mime: string;
  label: string;
} {
  if (summary.kind !== "table")
    return {
      name: "data.json",
      mime: "application/json",
      label: "Download JSON",
    };
  return summary.delimiter === "\t"
    ? {
        name: "table.tsv",
        mime: "text/tab-separated-values",
        label: "Download TSV",
      }
    : { name: "table.csv", mime: "text/csv", label: "Download CSV" };
}

function summarizeJSON(text: string): DataSummary | null {
  if (!/^[[{]/.test(text)) return null;
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  // Text opening with { or [ that parses is always an object or a list.
  if (Array.isArray(value)) return { kind: "json-list", items: value.length };
  const entries = Object.entries(value as object);
  return {
    kind: "json",
    fields: entries.map(([key]) => key),
    sample: entries
      .filter(
        ([key, item]) =>
          ["string", "number", "boolean"].includes(typeof item) &&
          !looksTechnical(key) &&
          !looksTechnical(String(item)),
      )
      .slice(0, SAMPLE_FIELDS)
      .map(([key, item]) => [key, String(item)]),
  };
}

/** A header and at least two rows, every line splitting into the same number
 *  (two or more) of cells. Prose with a stray comma per line fails one of the
 *  checks: too few lines, every line ending like a sentence, or "column
 *  names" that are long or sentence-like. */
function summarizeTable(text: string): DataSummary | null {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < MIN_TABLE_LINES) return null;
  if (lines.every((line) => /[.!?]\s*$/.test(line))) return null;
  const delimiter = lines[0].includes("\t") ? "\t" : ",";
  const rows = lines.map((line) => splitRow(line, delimiter));
  const width = rows[0].length;
  if (width < 2 || rows.some((cells) => cells.length !== width)) return null;
  const header = rows[0].map((cell) => cell.trim());
  const looksLikeHeader = header.every(
    (cell) => cell && cell.length <= MAX_HEADER_CELL && !/[.!?]\s/.test(cell),
  );
  if (!looksLikeHeader) return null;
  return {
    kind: "table",
    rows: rows.length - 1,
    columns: header,
    delimiter,
    sample: rows
      .slice(1, 1 + SAMPLE_ROWS)
      .map((cells) => cells.map((cell) => cell.trim())),
  };
}

function splitRow(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (const char of line) {
    if (char === '"') quoted = !quoted;
    else if (char === delimiter && !quoted) {
      cells.push(cell);
      cell = "";
    } else cell += char;
  }
  cells.push(cell);
  return cells;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** The first few names, leaving out any that look like a path or code:
 *  those stay behind technical details. With nothing plain left, a count. */
function listPreview(names: string[], noun: string): string {
  const plain = names.filter((name) => !looksTechnical(name));
  if (plain.length === 0) return plural(names.length, noun);
  const shown = plain.slice(0, 4).join(", ");
  return plain.length > 4 ? `${shown}, …` : shown;
}
