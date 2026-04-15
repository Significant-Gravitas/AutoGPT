import React, { useMemo, useState } from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

function normalizeMime(mime?: string): string | undefined {
  return mime?.toLowerCase().split(";")[0]?.trim();
}

function getDelimiter(metadata?: OutputMetadata): "," | "\t" {
  if (
    normalizeMime(metadata?.mimeType) === "text/tab-separated-values" ||
    metadata?.filename?.toLowerCase().endsWith(".tsv")
  ) {
    return "\t";
  }

  return ",";
}

function getDelimitedMimeType(metadata?: OutputMetadata): string {
  return getDelimiter(metadata) === "\t"
    ? "text/tab-separated-values"
    : "text/csv";
}

function getDelimitedFallbackFilename(metadata?: OutputMetadata): string {
  return getDelimiter(metadata) === "\t" ? "data.tsv" : "data.csv";
}

function parseDelimitedText(
  text: string,
  delimiter: "," | "\t",
): { headers: string[]; rows: string[][] } {
  const normalized = text
    .replace(/\r\n?/g, "\n")
    .replace(/^\ufeff/, "")
    .trim();
  if (normalized.length === 0) return { headers: [], rows: [] };

  // Character-by-character parse so embedded newlines inside "quoted" cells
  // (allowed by RFC 4180) don't break the row split.
  const rows: string[][] = [];
  let current = "";
  let row: string[] = [];
  let inQuotes = false;
  for (let i = 0; i < normalized.length; i++) {
    const ch = normalized[i];
    if (inQuotes) {
      if (ch === '"' && normalized[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === delimiter) {
      row.push(current);
      current = "";
    } else if (ch === "\n") {
      row.push(current);
      rows.push(row);
      row = [];
      current = "";
    } else {
      current += ch;
    }
  }
  row.push(current);
  rows.push(row);

  const headers = rows[0] ?? [];
  return { headers, rows: rows.slice(1) };
}

function CSVTable({
  value,
  delimiter,
}: {
  value: string;
  delimiter: "," | "\t";
}) {
  const { headers, rows } = useMemo(
    () => parseDelimitedText(value, delimiter),
    [delimiter, value],
  );
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  const sortedRows = useMemo(() => {
    if (sortCol === null) return rows;
    return [...rows].sort((a, b) => {
      const aVal = a[sortCol] ?? "";
      const bVal = b[sortCol] ?? "";
      const aNum = parseFloat(aVal);
      const bNum = parseFloat(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return sortAsc ? aNum - bNum : bNum - aNum;
      }
      return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
  }, [rows, sortCol, sortAsc]);

  function handleSort(col: number) {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  }

  if (headers.length === 0) {
    return <p className="p-4 text-sm text-zinc-500">Empty CSV</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-zinc-200 bg-zinc-50">
            {headers.map((header, i) => (
              <th
                key={i}
                className="px-3 py-2 text-left font-medium text-zinc-700"
              >
                <button
                  type="button"
                  className="flex w-full cursor-pointer select-none items-center gap-1 hover:bg-zinc-100"
                  onClick={() => handleSort(i)}
                >
                  {header}
                  {sortCol === i && (
                    <span className="text-xs">
                      {sortAsc ? "\u25B2" : "\u25BC"}
                    </span>
                  )}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, rowIdx) => (
            <tr
              key={rowIdx}
              className="border-b border-zinc-100 even:bg-zinc-50/50"
              style={{
                contentVisibility: "auto",
                containIntrinsicSize: "0 36px",
              }}
            >
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} className="px-3 py-1.5 text-zinc-600">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function canRenderCSV(value: unknown, metadata?: OutputMetadata): boolean {
  if (typeof value !== "string") return false;
  const mime = normalizeMime(metadata?.mimeType);
  if (mime === "text/csv" || mime === "text/tab-separated-values") {
    return true;
  }
  if (metadata?.filename?.toLowerCase().endsWith(".csv")) return true;
  if (metadata?.filename?.toLowerCase().endsWith(".tsv")) return true;
  return false;
}

function renderCSV(value: unknown, metadata?: OutputMetadata): React.ReactNode {
  return <CSVTable value={String(value)} delimiter={getDelimiter(metadata)} />;
}

function getCopyContentCSV(
  value: unknown,
  _metadata?: OutputMetadata,
): CopyContent | null {
  const text = String(value);
  return { mimeType: "text/plain", data: text, fallbackText: text };
}

function getDownloadContentCSV(
  value: unknown,
  metadata?: OutputMetadata,
): DownloadContent | null {
  const text = String(value);
  const mimeType = getDelimitedMimeType(metadata);
  return {
    data: new Blob([text], { type: mimeType }),
    filename: metadata?.filename || getDelimitedFallbackFilename(metadata),
    mimeType,
  };
}

export const csvRenderer: OutputRenderer = {
  name: "CSVRenderer",
  priority: 38,
  canRender: canRenderCSV,
  render: renderCSV,
  getCopyContent: getCopyContentCSV,
  getDownloadContent: getDownloadContentCSV,
  isConcatenable: () => false,
};
