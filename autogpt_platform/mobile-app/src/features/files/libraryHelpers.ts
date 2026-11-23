// Relative imports: jest has no `@/` alias.
import type { LibraryFilter, StorageUsage, WorkspaceFolder } from "../../lib/api/workspace";
import { formatFileSize } from "./listHelpers";

/** The Files page's chips: everything, what you uploaded, or one expert's work. */
export type OwnerChip = { kind: "all" } | { kind: "you" } | { kind: "expert"; id: string };

export function libraryFilter(chip: OwnerChip, folderID: string | null, search: string): LibraryFilter {
  const q = search.trim();
  return {
    q: q || null,
    origin: chip.kind === "you" ? "uploaded" : null,
    expertID: chip.kind === "expert" ? chip.id : null,
    folderID,
  };
}

export function childFolders(folders: WorkspaceFolder[], parentID: string | null) {
  return folders
    .filter((folder) => folder.parent_id === parentID)
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** The folders from the top down to this one, this one last. Empty when it isn't there. */
export function folderPath(folders: WorkspaceFolder[], folderID: string) {
  const byID = new Map(folders.map((folder) => [folder.id, folder]));
  const path: WorkspaceFolder[] = [];
  let current = byID.get(folderID);
  // A cycle in bad data would loop forever; a path can't be longer than the list.
  while (current && path.length <= folders.length) {
    path.unshift(current);
    current = current.parent_id ? byID.get(current.parent_id) : undefined;
  }
  return path;
}

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** "5 files", "4 files · 1 folder". */
export function folderSummary(folder: WorkspaceFolder, folders: WorkspaceFolder[]) {
  const subfolders = folders.filter((other) => other.parent_id === folder.id).length;
  const parts = [plural(folder.file_count, "file")];
  if (subfolders > 0) parts.push(plural(subfolders, "folder"));
  return parts.join(" · ");
}

export function storageLine(usage: StorageUsage) {
  return `${formatFileSize(usage.used_bytes)} of ${formatFileSize(usage.limit_bytes)}`;
}

/** Rows and cells of a CSV or TSV, with quoted cells as RFC 4180 has them. */
export function parseDelimited(text: string, delimiter: "," | "\t") {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (inQuotes) {
      if (char === '"' && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        cell += char;
      }
    } else if (char === '"' && cell === "") {
      inQuotes = true;
    } else if (char === delimiter) {
      row.push(cell);
      cell = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell !== "" || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

// Enough rows to read on a phone without building thousands of views.
export const TABLE_ROW_LIMIT = 500;
const CHAR_WIDTH = 7.5;
const MIN_COLUMN = 64;
const MAX_COLUMN = 240;

export interface TableData {
  header: string[];
  rows: string[][];
  totalRows: number;
  widths: number[];
  numeric: boolean[];
}

/** A CSV or TSV ready for the table viewer: the first row as headings, columns sized to their text. */
export function tableFromText(text: string, name: string): TableData {
  const delimiter = /\.tsv$/i.test(name) ? "\t" : ",";
  const [header = [], ...body] = parseDelimited(text, delimiter).filter(
    (row) => !(row.length === 1 && row[0] === ""),
  );
  const columns = Math.max(header.length, ...body.map((row) => row.length));
  const rows = body.slice(0, TABLE_ROW_LIMIT);
  const widths = Array.from({ length: columns }, (_, column) => {
    const longest = Math.max(
      (header[column] ?? "").length,
      ...rows.map((row) => (row[column] ?? "").length),
    );
    return Math.min(MAX_COLUMN, Math.max(MIN_COLUMN, Math.ceil(longest * CHAR_WIDTH) + 24));
  });
  const numeric = Array.from({ length: columns }, (_, column) => {
    const cells = rows.map((row) => (row[column] ?? "").trim()).filter(Boolean);
    return cells.length > 0 && cells.every((cell) => /^[-+$€£]?[\d,.]+%?$/.test(cell));
  });
  return { header, rows, totalRows: body.length, widths, numeric };
}

/** JSON laid out two spaces deep; anything that doesn't parse comes back as it was. */
export function formatJson(text: string) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}
