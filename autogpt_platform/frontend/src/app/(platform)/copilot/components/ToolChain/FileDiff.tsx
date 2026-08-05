"use client";

import { CodeIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { MAX_RENDERED_DIFF_ROWS, parseUnifiedDiff } from "./fileDiffHelpers";

interface Props {
  file?: string;
  diff: string;
}

const LN_CLASSES = "select-none px-[7px] text-right text-[11px] leading-5";

export function FileDiff({ file, diff }: Props) {
  const { rows, added, removed, truncated } = parseUnifiedDiff(diff);

  return (
    <div className="overflow-hidden rounded-xl bg-white font-mono shadow-sm ring-1 ring-zinc-200/70">
      <div className="flex items-center gap-2 border-b border-zinc-200 py-2.5 pl-4 pr-3 text-[12.5px]">
        <span className="inline-flex items-center gap-[7px]">
          <Icon icon={CodeIcon} size={15} className="shrink-0 text-zinc-400" />
          <span className="leading-none text-zinc-900">{file ?? "file"}</span>
        </span>
        <span className="ml-auto inline-flex items-center gap-2 text-xs leading-none">
          <span className="text-green-600">+{added}</span>
          <span className="text-red-600">-{removed}</span>
        </span>
      </div>
      <div className="relative max-h-64 overflow-y-auto py-1 text-[12.5px] leading-5 scrollbar-none before:absolute before:bottom-0 before:left-16 before:top-0 before:z-10 before:w-px before:bg-zinc-200 before:content-['']">
        {rows.map((row) => (
          <div
            key={`${row.type}:${row.old ?? ""}:${row.cur ?? ""}`}
            className={
              "relative grid grid-cols-[32px_32px_18px_1fr] items-stretch " +
              (row.type === "add"
                ? "bg-green-50"
                : row.type === "del"
                  ? "bg-red-50"
                  : "")
            }
          >
            {row.type === "add" && (
              <span className="absolute inset-y-0 left-0 w-[3px] bg-green-600" />
            )}
            {row.type === "del" && (
              <span className="absolute inset-y-0 left-0 w-[3px] bg-red-600" />
            )}
            <span
              className={
                LN_CLASSES +
                (row.type === "del" ? " text-red-600" : " text-zinc-400")
              }
            >
              {row.old ?? ""}
            </span>
            <span
              className={
                LN_CLASSES +
                (row.type === "add" ? " text-green-600" : " text-zinc-400")
              }
            >
              {row.cur ?? ""}
            </span>
            <span
              className={
                "select-none text-center text-[11px] leading-5 " +
                (row.type === "add"
                  ? "text-green-600"
                  : row.type === "del"
                    ? "text-red-600"
                    : "text-zinc-400")
              }
            >
              {row.type === "add" ? "+" : row.type === "del" ? "-" : ""}
            </span>
            <code
              className={
                "whitespace-pre pl-2 pr-3 " +
                (row.type === "ctx" ? "text-zinc-400" : "text-zinc-900")
              }
            >
              {row.text}
            </code>
          </div>
        ))}
        {truncated && (
          <p className="px-3 py-2 text-center text-xs text-zinc-500">
            Diff preview truncated after {MAX_RENDERED_DIFF_ROWS} lines
          </p>
        )}
      </div>
    </div>
  );
}
