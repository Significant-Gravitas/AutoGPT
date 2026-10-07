"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { forwardRef, ReactElement, ReactNode, Ref } from "react";
import { SortableHeader } from "./components/SortableHeader";
import {
  alignClassName,
  DEFAULT_ALIGN,
  DataTableColumn,
  DataTableSort,
  getAriaSort,
} from "./helpers";
import { useDataTable } from "./useDataTable";

interface Props<T> {
  columns: DataTableColumn<T>[];
  rows: T[];
  getRowKey: (row: T, index: number) => string;
  /** Accessible table name, rendered as an sr-only caption. */
  caption: string;
  isLoading?: boolean;
  loadingRowCount?: number;
  emptyState?: ReactNode;
  onRowClick?: (row: T) => void;
  /** Controlled sort. Leave undefined to let the table manage it. */
  sort?: DataTableSort | null;
  defaultSort?: DataTableSort | null;
  onSortChange?: (sort: DataTableSort | null) => void;
  /** The rows arrive sorted (e.g. by the server); only report sort changes. */
  manualSorting?: boolean;
  className?: string;
}

function DataTableInner<T>(
  {
    columns,
    rows,
    getRowKey,
    caption,
    isLoading = false,
    loadingRowCount = 5,
    emptyState = "No results",
    onRowClick,
    sort,
    defaultSort,
    onSortChange,
    manualSorting = false,
    className,
  }: Props<T>,
  ref: Ref<HTMLTableElement>,
) {
  const { activeSort, visibleRows, handleSort } = useDataTable({
    rows,
    columns,
    sort,
    defaultSort,
    onSortChange,
    manualSorting,
  });

  function handleRowKeyDown(event: React.KeyboardEvent, row: T) {
    if (event.target !== event.currentTarget) return;
    if (isKey(event, "Enter", " ")) {
      event.preventDefault();
      onRowClick?.(row);
    }
  }

  function renderBody() {
    if (isLoading) {
      return Array.from({ length: loadingRowCount }, (_, rowIndex) => (
        <tr
          key={`loading-${rowIndex}`}
          className="border-b border-border last:border-0"
        >
          {columns.map((column) => (
            <td key={column.key} className="px-4 py-3">
              <Skeleton className="h-4 w-full max-w-32" />
            </td>
          ))}
        </tr>
      ));
    }

    if (visibleRows.length === 0) {
      return (
        <tr>
          <td colSpan={columns.length} className="px-4 py-10 text-center">
            {typeof emptyState === "string" ? (
              <Text variant="body" tone="secondary" as="span">
                {emptyState}
              </Text>
            ) : (
              emptyState
            )}
          </td>
        </tr>
      );
    }

    return visibleRows.map((row, rowIndex) => (
      <tr
        key={getRowKey(row, rowIndex)}
        onClick={onRowClick ? () => onRowClick(row) : undefined}
        onKeyDown={
          onRowClick ? (event) => handleRowKeyDown(event, row) : undefined
        }
        tabIndex={onRowClick ? 0 : undefined}
        className={cn(
          "border-b border-border transition-colors last:border-0",
          onRowClick &&
            "cursor-pointer focus-ring hover:bg-muted/50 focus-visible:bg-muted/50 focus-visible:ring-inset",
        )}
      >
        {columns.map((column) => (
          <td
            key={column.key}
            className={cn(
              "px-4 py-3 align-middle font-sans text-sm text-foreground",
              alignClassName[column.align ?? DEFAULT_ALIGN],
              column.className,
            )}
          >
            {column.cell(row, rowIndex)}
          </td>
        ))}
      </tr>
    ));
  }

  return (
    <div
      className={cn(
        "relative w-full overflow-x-auto rounded-xl border border-border bg-card",
        className,
      )}
    >
      <table
        ref={ref}
        aria-busy={isLoading || undefined}
        className="w-full caption-bottom border-collapse"
      >
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-muted/50">
          <tr className="border-b border-border">
            {columns.map((column) => {
              const align = column.align ?? DEFAULT_ALIGN;
              const isSortable = Boolean(column.sortValue);
              return (
                <th
                  key={column.key}
                  scope="col"
                  aria-sort={
                    isSortable ? getAriaSort(activeSort, column.key) : undefined
                  }
                  className={cn(
                    "h-10 px-4 align-middle",
                    alignClassName[align],
                    column.headerClassName,
                  )}
                >
                  {isSortable ? (
                    <SortableHeader
                      align={align}
                      direction={
                        activeSort?.key === column.key
                          ? activeSort.direction
                          : null
                      }
                      onSort={() => handleSort(column.key)}
                    >
                      {column.header}
                    </SortableHeader>
                  ) : (
                    <Text variant="small-medium" as="span" tone="secondary">
                      {column.header}
                    </Text>
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>{renderBody()}</tbody>
      </table>
    </div>
  );
}

// forwardRef drops the row type parameter, so restore it on the export.
export const DataTable = forwardRef(DataTableInner) as <T>(
  props: Props<T> & { ref?: Ref<HTMLTableElement> },
) => ReactElement;
