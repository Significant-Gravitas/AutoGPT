"use client";

import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { Select } from "@/components/atoms/Select/Select";
import { cn } from "@/lib/utils";
import { statusDotClass } from "../../helpers";

interface Props {
  sessionId: string;
  columns: BoardColumn[];
  columnKey: string | null;
  onMove: (column: string) => void;
}

export function StatusSelect({ sessionId, columns, columnKey, onMove }: Props) {
  return (
    <div className="relative">
      <span
        aria-hidden
        className={cn(
          "pointer-events-none absolute left-2.5 top-1/2 z-10 size-2 -translate-y-1/2 rounded-full",
          statusDotClass(columnKey),
        )}
      />
      <Select
        id={`task-column-${sessionId}`}
        label="Column"
        hideLabel
        size="small"
        placeholder="Move to…"
        value={columnKey ?? undefined}
        onValueChange={onMove}
        options={columns.map((column) => ({
          value: column.key,
          label: column.name,
        }))}
        wrapperClassName="mb-0"
        className="h-7 w-auto min-w-0 gap-1.5 rounded-full border-zinc-200 bg-zinc-50 pl-6 pr-2 text-xs font-medium text-zinc-700 shadow-none hover:bg-zinc-100"
      />
    </div>
  );
}
