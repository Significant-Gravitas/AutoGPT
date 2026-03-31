import React, { useMemo, useState } from "react";
import {
  OutputRenderer,
  OutputMetadata,
  DownloadContent,
  CopyContent,
} from "../types";

function parseCSV(text: string): { headers: string[]; rows: string[][] } {
  const lines = text.trim().split("\n");
  if (lines.length === 0) return { headers: [], rows: [] };

  const parseLine = (line: string): string[] => {
    const result: string[] = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (inQuotes) {
        if (ch === '"' && line[i + 1] === '"') {
          current += '"';
          i++;
        } else if (ch === '"') {
          inQuotes = false;
        } else {
          current += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === ",") {
          result.push(current);
          current = "";
        } else {
          current += ch;
        }
      }
    }
    result.push(current);
    return result;
  };

  const headers = parseLine(lines[0]);
  const rows = lines.slice(1).map(parseLine);
  return { headers, rows };
}

function CSVTable({ value }: { value: string }) {
  const { headers, rows } = useMemo(() => parseCSV(value), [value]);
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
                className="cursor-pointer select-none px-3 py-2 text-left font-medium text-zinc-700 hover:bg-zinc-100"
                onClick={() => handleSort(i)}
              >
                <span className="flex items-center gap-1">
                  {header}
                  {sortCol === i && (
                    <span className="text-xs">
                      {sortAsc ? "\u25B2" : "\u25BC"}
                    </span>
                  )}
                </span>
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
  if (metadata?.mimeType === "text/csv") return true;
  if (metadata?.filename?.toLowerCase().endsWith(".csv")) return true;
  return false;
}

function renderCSV(
  value: unknown,
  _metadata?: OutputMetadata,
): React.ReactNode {
  return <CSVTable value={String(value)} />;
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
  return {
    data: new Blob([text], { type: "text/csv" }),
    filename: metadata?.filename || "data.csv",
    mimeType: "text/csv",
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
