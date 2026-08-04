"use client";

import { CodeIcon } from "@phosphor-icons/react";

interface DiffRow {
  old: number | null;
  cur: number | null;
  type: "ctx" | "add" | "del";
  text: string;
}

export function isDiffText(value: unknown): value is string {
  if (typeof value !== "string") return false;
  return /^@@/m.test(value) || (/^\+/m.test(value) && /^-/m.test(value));
}

function parseUnifiedDiff(diff: string): DiffRow[] {
  const rows: DiffRow[] = [];
  let oldLn = 1;
  let newLn = 1;
  for (const line of diff.split("\n")) {
    const hunk = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunk) {
      oldLn = Number(hunk[1]);
      newLn = Number(hunk[2]);
      continue;
    }
    if (line.startsWith("-")) {
      rows.push({ old: oldLn++, cur: null, type: "del", text: line.slice(1) });
    } else if (line.startsWith("+")) {
      rows.push({ old: null, cur: newLn++, type: "add", text: line.slice(1) });
    } else {
      rows.push({
        old: oldLn++,
        cur: newLn++,
        type: "ctx",
        text: line.startsWith(" ") ? line.slice(1) : line,
      });
    }
  }
  return rows;
}

const LN_CLASSES = "select-none px-[7px] text-right text-[11px] leading-5";

export function FileDiff({ file, diff }: { file?: string; diff: string }) {
  const rows = parseUnifiedDiff(diff);
  const added = rows.filter((r) => r.type === "add").length;
  const removed = rows.filter((r) => r.type === "del").length;

  return (
    <div className="overflow-hidden rounded-xl bg-white font-mono shadow-[0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02),0_0_0_0.5px_rgba(0,0,0,0.08)]">
      <div className="flex items-center gap-2 border-b border-[#e6e8ec] py-2.5 pl-4 pr-3 text-[12.5px]">
        <span className="inline-flex items-center gap-[7px]">
          <CodeIcon size={15} className="shrink-0 text-zinc-400" />
          <span className="leading-none text-zinc-900">{file ?? "file"}</span>
        </span>
        <span className="ml-auto inline-flex items-center gap-2 text-xs leading-none">
          <span className="text-[#15a06a]">+{added}</span>
          <span className="text-[#dc2626]">-{removed}</span>
        </span>
      </div>
      <div className="relative max-h-64 overflow-y-auto py-1 text-[12.5px] leading-5 scrollbar-none before:absolute before:bottom-0 before:left-16 before:top-0 before:z-10 before:w-px before:bg-[#e6e8eb] before:content-['']">
        {rows.map((row, i) => (
          <div
            key={i}
            className={
              "relative grid grid-cols-[32px_32px_18px_1fr] items-stretch " +
              (row.type === "add"
                ? "bg-[rgba(26,127,55,0.09)]"
                : row.type === "del"
                  ? "bg-[rgba(207,34,46,0.09)]"
                  : "")
            }
          >
            {row.type === "add" && (
              <span className="absolute inset-y-0 left-0 w-[3px] bg-[#15a06a]" />
            )}
            {row.type === "del" && (
              <span className="absolute inset-y-0 left-0 w-[3px] bg-[repeating-linear-gradient(45deg,#dc2626_0,#dc2626_1.5px,transparent_1.5px,transparent_3px)]" />
            )}
            <span
              className={
                LN_CLASSES +
                (row.type === "del" ? " text-[#dc2626]" : " text-zinc-400")
              }
            >
              {row.old ?? ""}
            </span>
            <span
              className={
                LN_CLASSES +
                (row.type === "add" ? " text-[#15a06a]" : " text-zinc-400")
              }
            >
              {row.cur ?? ""}
            </span>
            <span
              className={
                "select-none text-center text-[11px] leading-5 " +
                (row.type === "add"
                  ? "text-[#15a06a]"
                  : row.type === "del"
                    ? "text-[#dc2626]"
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
      </div>
    </div>
  );
}
