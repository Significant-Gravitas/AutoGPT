"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { getFilePreviewUrl } from "../helpers";
import { parseCsv, type CsvPreview } from "../parsers";
import { LoadingPlaceholder, useFileText } from "./PreviewParts";

const TEXT_SNIPPET_CHARS = 1500;

interface PreviewProps {
  file: WorkspaceFileItem;
  onError: () => void;
}

const preClass =
  "h-full w-full overflow-hidden whitespace-pre-wrap break-words bg-white p-3 font-mono text-[10px] leading-[1.35] text-zinc-700";

export function TextSnippetPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: 2048 }),
    onError,
  );

  if (text === null) return <LoadingPlaceholder file={file} />;
  return <pre className={preClass}>{text.slice(0, TEXT_SNIPPET_CHARS)}</pre>;
}

export function JsonPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: 4096 }),
    onError,
  );

  if (text === null) return <LoadingPlaceholder file={file} />;
  return (
    <pre className={preClass}>
      {prettyJson(text).slice(0, TEXT_SNIPPET_CHARS)}
    </pre>
  );
}

function prettyJson(text: string): string {
  try {
    // The byte-capped fetch often truncates JSON mid-value, so this only
    // succeeds for small files; otherwise show the raw text.
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

export function CsvPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: 4096 }),
    onError,
  );

  if (text === null) return <LoadingPlaceholder file={file} />;

  const parsed = parseCsv(text);
  if (!parsed) {
    return <pre className={preClass}>{text.slice(0, TEXT_SNIPPET_CHARS)}</pre>;
  }
  return <CsvTable preview={parsed} />;
}

function CsvTable({ preview }: { preview: CsvPreview }) {
  const cellClass =
    "max-w-[6rem] truncate border-b border-zinc-100 px-2 py-1 text-left";
  return (
    <div className="relative h-full w-full overflow-hidden bg-white">
      <table className="w-full table-fixed border-collapse text-[9px] text-zinc-700">
        <thead>
          <tr>
            {preview.headers.map((header, i) => (
              <th
                key={i}
                className={`${cellClass} bg-zinc-50 font-semibold text-zinc-900`}
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c} className={cellClass}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-white to-transparent" />
    </div>
  );
}
