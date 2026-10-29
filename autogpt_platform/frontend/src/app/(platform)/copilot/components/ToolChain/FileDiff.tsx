"use client";

import { CodeIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { MAX_RENDERED_DIFF_ROWS, parseUnifiedDiff } from "./fileDiffHelpers";

interface Props {
  file?: string;
  diff: string;
}

const LN_CLASSES = "select-none px-1.5 text-right text-caption-sm leading-5";

export function FileDiff({ file, diff }: Props) {
  const { rows, added, removed, truncated } = parseUnifiedDiff(diff);

  return (
    <div className="overflow-hidden rounded-xl bg-surface font-mono shadow-sm ring-1 ring-border-base/70">
      <div className="flex items-center gap-2 border-b border-border-base py-2.5 pl-4 pr-3 text-body-sm leading-inherit">
        <span className="inline-flex items-center gap-1.5">
          <Icon
            icon={CodeIcon}
            size={15}
            className="shrink-0 text-foreground-subtle"
          />
          <span className="leading-none text-foreground-strong">
            {file ?? "file"}
          </span>
        </span>
        <span className="ml-auto inline-flex items-center gap-2 text-xs leading-none">
          <span className="text-success">+{added}</span>
          <span className="text-error-foreground">-{removed}</span>
        </span>
      </div>
      <div className="relative max-h-64 overflow-y-auto py-1 text-body-sm leading-5 scrollbar-none before:absolute before:bottom-0 before:left-16 before:top-0 before:z-10 before:w-px before:bg-surface-active before:content-['']">
        {rows.map((row) => (
          <div
            key={`${row.type}:${row.old ?? ""}:${row.cur ?? ""}`}
            className={
              "relative grid grid-cols-[32px_32px_18px_1fr] items-stretch " +
              (row.type === "add"
                ? "bg-success-surface"
                : row.type === "del"
                  ? "bg-error-surface"
                  : "")
            }
          >
            {row.type === "add" && (
              <span className="absolute inset-y-0 left-0 w-[3px] bg-success" />
            )}
            {row.type === "del" && (
              <span className="absolute inset-y-0 left-0 w-[3px] bg-error-hover" />
            )}
            <span
              className={
                LN_CLASSES +
                (row.type === "del"
                  ? " text-error-foreground"
                  : " text-foreground-subtle")
              }
            >
              {row.old ?? ""}
            </span>
            <span
              className={
                LN_CLASSES +
                (row.type === "add"
                  ? " text-success"
                  : " text-foreground-subtle")
              }
            >
              {row.cur ?? ""}
            </span>
            <span
              className={
                "select-none text-center text-caption-sm leading-5 " +
                (row.type === "add"
                  ? "text-success"
                  : row.type === "del"
                    ? "text-error-foreground"
                    : "text-foreground-subtle")
              }
            >
              {row.type === "add" ? "+" : row.type === "del" ? "-" : ""}
            </span>
            <code
              className={
                "whitespace-pre pl-2 pr-3 " +
                (row.type === "ctx"
                  ? "text-foreground-subtle"
                  : "text-foreground-strong")
              }
            >
              {row.text}
            </code>
          </div>
        ))}
        {truncated && (
          <p className="px-3 py-2 text-center text-xs text-foreground-tertiary">
            Diff preview truncated after {MAX_RENDERED_DIFF_ROWS} lines
          </p>
        )}
      </div>
    </div>
  );
}
