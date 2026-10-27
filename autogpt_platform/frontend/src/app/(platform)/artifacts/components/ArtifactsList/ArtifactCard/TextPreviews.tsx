"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { getFilePreviewUrl } from "../helpers";
import { parseCsv, type CsvPreview } from "../parsers";
import { LoadingPlaceholder, useFileText } from "./PreviewParts";

const TEXT_SNIPPET_CHARS = 1500;

interface PreviewProps {
  file: WorkspaceFileItem;
  onError: () => void;
}

const preClass =
  // TODO(tokens): no exact text token for leading-[1.35]
  "h-full w-full overflow-hidden whitespace-pre-wrap break-words bg-surface p-3 font-mono text-tiny leading-[1.35] text-foreground-body";

// Fetch a larger slice and show more rows than the default so the card table
// fills the preview area; the container clips overflow under the bottom fade.
const CSV_PREVIEW_BYTES = 16384;
const CSV_PREVIEW_MAX_ROWS = 20;

export function CsvPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: CSV_PREVIEW_BYTES }),
    onError,
  );

  if (text === null) return <LoadingPlaceholder file={file} />;

  const parsed = parseCsv(text, { maxRows: CSV_PREVIEW_MAX_ROWS });
  if (!parsed) {
    return <pre className={preClass}>{text.slice(0, TEXT_SNIPPET_CHARS)}</pre>;
  }
  return <CsvTable preview={parsed} />;
}

// Compact element map so rendered markdown stays legible inside the small
// card preview — no typography plugin is installed, so default element sizes
// (huge h1, etc.) would overflow without these overrides.
const MARKDOWN_COMPONENTS: Components = {
  h1: ({ children }) => (
    <p className="mb-1 text-caption-sm font-semibold leading-inherit text-foreground-strong">
      {children}
    </p>
  ),
  h2: ({ children }) => (
    <p className="mb-1 text-caption-sm font-semibold leading-inherit text-foreground-strong">
      {children}
    </p>
  ),
  h3: ({ children }) => (
    <p className="mb-0.5 text-tiny font-semibold leading-inherit text-foreground-emphasis">
      {children}
    </p>
  ),
  p: ({ children }) => (
    <p
      // TODO(tokens): no exact text token for leading-[1.4]
      className="mb-1 text-tiny leading-[1.4] text-foreground-secondary"
    >
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul className="mb-1 ml-3 list-disc text-tiny leading-inherit text-foreground-secondary">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mb-1 ml-3 list-decimal text-tiny leading-inherit text-foreground-secondary">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="mb-0.5">{children}</li>,
  // TODO(tokens): no exact role for text-violet-600 (#7C3AED)
  a: ({ children }) => <span className="text-violet-600">{children}</span>,
  code: ({ children }) => (
    <code
      // TODO(tokens): SECRT-3152, rare size text-[9px]
      className="rounded bg-surface-hover px-1 font-mono text-[9px]"
    >
      {children}
    </code>
  ),
  pre: ({ children }) => (
    <pre
      // TODO(tokens): SECRT-3152, rare size text-[9px]
      className="mb-1 overflow-hidden rounded bg-surface-hover p-1.5 font-mono text-[9px] text-foreground-body"
    >
      {children}
    </pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mb-1 border-l-2 border-border-base pl-2 text-tiny leading-inherit text-foreground-tertiary">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-1 border-separator" />,
  img: () => null,
};

export function MarkdownPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: 4096 }),
    onError,
  );

  if (text === null) return <LoadingPlaceholder file={file} />;

  return (
    <div className="relative h-full w-full overflow-hidden bg-surface">
      <div className="h-full w-full overflow-hidden p-3">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={MARKDOWN_COMPONENTS}
        >
          {text.slice(0, TEXT_SNIPPET_CHARS)}
        </ReactMarkdown>
      </div>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-white to-transparent" />
    </div>
  );
}

function CsvTable({ preview }: { preview: CsvPreview }) {
  const cellClass =
    "max-w-[6rem] truncate border-b border-separator px-2 py-1 text-left";
  return (
    <div className="relative h-full w-full overflow-hidden bg-surface">
      <table
        // TODO(tokens): SECRT-3152, rare size text-[9px]
        className="w-full table-fixed border-collapse text-[9px] text-foreground-body"
      >
        <thead>
          <tr>
            {preview.headers.map((header, i) => (
              <th
                key={i}
                className={`${cellClass} bg-surface-muted font-semibold text-foreground-strong`}
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
