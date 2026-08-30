"use client";

import { cn } from "@/lib/utils";
import { KeyboardEvent, ReactNode, useState } from "react";

export interface FilterTableColumn {
  key: string;
  label: string;
  /** CSS grid track for this column, e.g. "minmax(0,1.3fr)". */
  width?: string;
}

export interface FilterTableFilter {
  key: string;
  label: string;
  /** Chip dot color (any CSS color). Omit for a plain chip. */
  dot?: string;
}

export interface FilterTableRow {
  id: string;
  /** Which filter chip bucket this row belongs to. */
  filterKey: string;
  cells: Record<string, ReactNode>;
  onClick?: () => void;
}

export interface FilterTableProps {
  columns: FilterTableColumn[];
  rows: FilterTableRow[];
  /** Chips rendered after the automatic "All" chip; counts derive from rows. */
  filters?: FilterTableFilter[];
  allLabel?: string;
  emptyText?: string;
  ariaLabel?: string;
  className?: string;
}

const DEFAULT_TRACK = "minmax(0,1fr)";
const ALL_KEY = "all";

export function FilterTable({
  columns,
  rows,
  filters = [],
  allLabel = "All",
  emptyText = "Nothing here yet.",
  ariaLabel,
  className,
}: FilterTableProps) {
  const [activeFilter, setActiveFilter] = useState(ALL_KEY);
  const gridTemplateColumns = columns
    .map((column) => column.width ?? DEFAULT_TRACK)
    .join(" ");

  return (
    <div className={cn("w-full", className)}>
      {filters.length > 0 && rows.length > 0 ? (
        <FilterChips
          filters={[{ key: ALL_KEY, label: allLabel }, ...filters]}
          rows={rows}
          activeFilter={activeFilter}
          onSelect={setActiveFilter}
        />
      ) : null}

      <div
        role="table"
        aria-label={ariaLabel}
        className="overflow-x-auto rounded-2xl bg-white ring-1 ring-inset ring-zinc-200"
      >
        <div className="min-w-[420px]">
          <div
            role="row"
            className="grid border-b border-zinc-100 text-xs font-medium text-zinc-500 [&>*:not(:last-child)]:border-r [&>*:not(:last-child)]:border-zinc-100"
            style={{ gridTemplateColumns }}
          >
            {columns.map((column) => (
              <span key={column.key} role="columnheader" className="px-3 py-2">
                {column.label}
              </span>
            ))}
          </div>

          {rows.length === 0 ? (
            <p className="px-3 py-6 text-center text-sm text-zinc-500">
              {emptyText}
            </p>
          ) : (
            rows.map((row) => (
              <CollapsibleRow
                key={row.id}
                row={row}
                columns={columns}
                gridTemplateColumns={gridTemplateColumns}
                shown={
                  activeFilter === ALL_KEY || row.filterKey === activeFilter
                }
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

interface ChipsProps {
  filters: FilterTableFilter[];
  rows: FilterTableRow[];
  activeFilter: string;
  onSelect: (key: string) => void;
}

function FilterChips({ filters, rows, activeFilter, onSelect }: ChipsProps) {
  return (
    <div
      className="-mx-1 mb-2 flex items-center gap-1 overflow-x-auto px-1 py-1"
      style={{ scrollbarWidth: "none" }}
    >
      {filters.map((filter) => {
        const active = activeFilter === filter.key;
        const count =
          filter.key === ALL_KEY
            ? rows.length
            : rows.filter((row) => row.filterKey === filter.key).length;
        return (
          <button
            key={filter.key}
            type="button"
            aria-pressed={active}
            onClick={() => onSelect(filter.key)}
            className={cn(
              "flex h-7 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium transition-colors",
              active
                ? "bg-white text-zinc-900 shadow-sm ring-1 ring-inset ring-zinc-200"
                : "text-zinc-500 hover:bg-zinc-100",
            )}
          >
            {filter.dot ? (
              <span
                className="size-1.5 rounded-full"
                style={{ background: filter.dot }}
              />
            ) : null}
            {filter.label}
            <span
              className={cn(
                "rounded px-1 text-[10.5px] tabular-nums",
                active ? "bg-zinc-100 text-zinc-500" : "text-zinc-400",
              )}
            >
              {count}
            </span>
          </button>
        );
      })}
    </div>
  );
}

interface RowProps {
  row: FilterTableRow;
  columns: FilterTableColumn[];
  gridTemplateColumns: string;
  shown: boolean;
}

function CollapsibleRow({
  row,
  columns,
  gridTemplateColumns,
  shown,
}: RowProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (row.onClick && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      row.onClick();
    }
  }

  return (
    <div
      aria-hidden={!shown}
      className="grid transition-[grid-template-rows,opacity] duration-300 ease-out"
      style={{
        gridTemplateRows: shown ? "1fr" : "0fr",
        opacity: shown ? 1 : 0,
      }}
    >
      <div className="overflow-hidden">
        <div
          role="row"
          tabIndex={row.onClick && shown ? 0 : undefined}
          onClick={row.onClick}
          onKeyDown={handleKeyDown}
          className={cn(
            "grid border-b border-zinc-100 text-sm transition-colors last:border-b-0 [&>*:not(:last-child)]:border-r [&>*:not(:last-child)]:border-zinc-100",
            row.onClick && "cursor-pointer hover:bg-zinc-50",
          )}
          style={{ gridTemplateColumns }}
        >
          {columns.map((column) => (
            <span
              key={column.key}
              role="cell"
              className="flex min-w-0 items-center px-3 py-2.5"
            >
              {row.cells[column.key]}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
