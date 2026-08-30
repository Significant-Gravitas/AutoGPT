"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { InboxIcon, More02Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { KeyboardEvent, ReactNode, useState } from "react";

export interface FilterTableColumn {
  key: string;
  label: string;
  /** Header icon rendered before the label. */
  icon?: IconSvgElement;
  /** CSS grid track for this column, e.g. "minmax(0,1.3fr)". */
  width?: string;
}

export interface FilterTableFilter {
  key: string;
  label: string;
  /** Chip leading icon. Falls back to a dot when omitted. */
  icon?: IconSvgElement;
  /** Color for the chip icon/dot (any CSS color). Omit for a plain chip. */
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
  allIcon?: IconSvgElement;
  /** Chips shown on the strip, counting "All"; the rest move into a menu. */
  maxVisibleFilters?: number;
  emptyIcon?: IconSvgElement;
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
  allIcon,
  maxVisibleFilters = Number.POSITIVE_INFINITY,
  emptyText = "Nothing here yet.",
  emptyIcon = InboxIcon,
  ariaLabel,
  className,
}: FilterTableProps) {
  const [activeFilter, setActiveFilter] = useState(ALL_KEY);
  const gridTemplateColumns = columns
    .map((column) => column.width ?? DEFAULT_TRACK)
    .join(" ");
  const visibleCount =
    activeFilter === ALL_KEY
      ? rows.length
      : rows.filter((row) => row.filterKey === activeFilter).length;
  const activeLabel =
    filters.find((filter) => filter.key === activeFilter)?.label ?? allLabel;

  return (
    <div className={cn("w-full", className)}>
      {filters.length > 0 && rows.length > 0 ? (
        <FilterChips
          filters={[
            { key: ALL_KEY, label: allLabel, icon: allIcon },
            ...filters,
          ]}
          rows={rows}
          activeFilter={activeFilter}
          maxVisible={maxVisibleFilters}
          onSelect={setActiveFilter}
        />
      ) : null}

      <div
        role="table"
        aria-label={ariaLabel}
        className="overflow-hidden rounded-xl bg-zinc-100 text-[13px] shadow-[0_0_0_0.5px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.05),0_2px_4px_rgba(0,0,0,0.02)]"
      >
        {/* `overflow-x: auto` alone also turns the y axis scrollable, which the
            collapsing rows trip on — pin y shut and hide the x bar. */}
        <div className="overflow-x-auto overflow-y-hidden scrollbar-none">
          <div className="min-w-[420px]">
            <div
              role="row"
              className="grid font-medium text-zinc-800"
              style={{ gridTemplateColumns }}
            >
              {columns.map((column) => (
                <span
                  key={column.key}
                  role="columnheader"
                  className="flex items-center gap-1.5 px-3 py-[7px]"
                >
                  {column.icon ? <Icon icon={column.icon} size={14} /> : null}
                  {column.label}
                </span>
              ))}
            </div>

            {/* The body carries its own hairline and rides half a pixel past
                the shell's bottom edge so the two borders read as one. */}
            <div className="-mb-[0.5px] overflow-hidden rounded-t-xl border-[0.5px] border-zinc-200 bg-white">
              {visibleCount === 0 ? (
                <EmptyState
                  icon={emptyIcon}
                  text={
                    rows.length === 0
                      ? emptyText
                      : `Nothing under "${activeLabel}" right now.`
                  }
                />
              ) : null}

              {rows.map((row) => (
                <CollapsibleRow
                  key={row.id}
                  row={row}
                  columns={columns}
                  gridTemplateColumns={gridTemplateColumns}
                  shown={
                    activeFilter === ALL_KEY || row.filterKey === activeFilter
                  }
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ChipsProps {
  filters: FilterTableFilter[];
  rows: FilterTableRow[];
  activeFilter: string;
  maxVisible: number;
  onSelect: (key: string) => void;
}

function FilterChips({
  filters,
  rows,
  activeFilter,
  maxVisible,
  onSelect,
}: ChipsProps) {
  function countFor(key: string) {
    return key === ALL_KEY
      ? rows.length
      : rows.filter((row) => row.filterKey === key).length;
  }

  const visible = filters.slice(0, maxVisible);
  const overflow = filters.slice(maxVisible);
  // A pick made from the menu stays on the strip, so the current filter is
  // never hidden behind the "more" button.
  const promoted = overflow.find((filter) => filter.key === activeFilter);

  return (
    <div
      className="-mx-1 mb-2 flex items-center gap-1 overflow-x-auto px-1 py-1"
      style={{ scrollbarWidth: "none" }}
    >
      {[...visible, ...(promoted ? [promoted] : [])].map((filter) => (
        <FilterChip
          key={filter.key}
          filter={filter}
          count={countFor(filter.key)}
          active={activeFilter === filter.key}
          onSelect={onSelect}
        />
      ))}

      {overflow.length > 0 ? (
        <DropdownMenu>
          <DropdownMenuTrigger
            aria-label="More filters"
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 data-[state=open]:bg-zinc-100"
          >
            <Icon icon={More02Icon} size={14} />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[10rem]">
            {overflow.map((filter) => (
              <DropdownMenuItem
                key={filter.key}
                onSelect={() => onSelect(filter.key)}
                className="flex items-center gap-2 text-xs"
              >
                <ChipMarker filter={filter} active />
                <span className="flex-1">{filter.label}</span>
                <span className="tabular-nums text-zinc-400">
                  {countFor(filter.key)}
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}
    </div>
  );
}

interface ChipProps {
  filter: FilterTableFilter;
  count: number;
  active: boolean;
  onSelect: (key: string) => void;
}

function FilterChip({ filter, count, active, onSelect }: ChipProps) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={() => onSelect(filter.key)}
      className={cn(
        "flex h-7 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium transition-colors",
        active
          ? "bg-zinc-100 text-zinc-900"
          : "text-zinc-500 hover:bg-zinc-50",
      )}
    >
      <ChipMarker filter={filter} active={active} />
      {filter.label}
      <span
        className={cn(
          "rounded px-1 text-[10.5px] tabular-nums",
          active ? "bg-white text-zinc-500" : "text-zinc-400",
        )}
      >
        {count}
      </span>
    </button>
  );
}

interface EmptyProps {
  icon: IconSvgElement;
  text: string;
}

function EmptyState({ icon, text }: EmptyProps) {
  return (
    <div className="flex flex-col items-center gap-2 px-3 py-10 text-center">
      <span className="flex size-10 items-center justify-center rounded-full bg-zinc-50 ring-1 ring-inset ring-zinc-100">
        <Icon icon={icon} size={20} className="text-zinc-400" />
      </span>
      <p className="max-w-sm text-sm text-zinc-500">{text}</p>
    </div>
  );
}

interface MarkerProps {
  filter: FilterTableFilter;
  active: boolean;
}

/** Icons carry the status color once the chip is selected; unselected chips
 *  stay monochrome so the strip doesn't read as a rainbow. */
function ChipMarker({ filter, active }: MarkerProps) {
  if (filter.icon) {
    return (
      <Icon
        icon={filter.icon}
        size={14}
        style={active && filter.dot ? { color: filter.dot } : undefined}
      />
    );
  }

  if (!filter.dot) return null;

  return (
    <span className="size-1.5 rounded-full" style={{ background: filter.dot }} />
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
      className="grid border-b-[0.5px] border-zinc-200 transition-[grid-template-rows,opacity] duration-300 ease-out last:border-b-0"
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
            "grid transition-colors [&>*:not(:last-child)]:border-r-[0.5px] [&>*:not(:last-child)]:border-zinc-200",
            row.onClick && "cursor-pointer hover:bg-zinc-50",
          )}
          style={{ gridTemplateColumns }}
        >
          {columns.map((column) => (
            <span
              key={column.key}
              role="cell"
              className="flex min-w-0 items-center px-3 py-[9px] text-zinc-900"
            >
              {row.cells[column.key]}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
