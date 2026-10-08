"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { ReactElement, ReactNode, Ref } from "react";
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
  ref?: Ref<HTMLTableElement>;
}

export function DataTable<T>({
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
  ref,
}: Props<T>): ReactElement {
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
        <TableRow key={`loading-${rowIndex}`} className="hover:bg-transparent">
          {columns.map((column) => (
            <TableCell key={column.key}>
              <Skeleton className="h-4 w-full max-w-32" />
            </TableCell>
          ))}
        </TableRow>
      ));
    }

    if (visibleRows.length === 0) {
      return (
        <TableRow className="hover:bg-transparent">
          <TableCell
            colSpan={columns.length}
            className="h-auto py-10 text-center whitespace-normal"
          >
            {typeof emptyState === "string" ? (
              <Text variant="body" tone="secondary" as="span">
                {emptyState}
              </Text>
            ) : (
              emptyState
            )}
          </TableCell>
        </TableRow>
      );
    }

    return visibleRows.map((row, rowIndex) => (
      <TableRow
        key={getRowKey(row, rowIndex)}
        onClick={onRowClick ? () => onRowClick(row) : undefined}
        onKeyDown={
          onRowClick ? (event) => handleRowKeyDown(event, row) : undefined
        }
        tabIndex={onRowClick ? 0 : undefined}
        className={cn(
          !onRowClick && "hover:bg-transparent",
          onRowClick &&
            "cursor-pointer focus-ring focus-visible:bg-muted/50 focus-visible:ring-inset",
        )}
      >
        {columns.map((column) => (
          <TableCell
            key={column.key}
            className={cn(
              "h-auto py-3 whitespace-normal",
              alignClassName[column.align ?? DEFAULT_ALIGN],
              column.className,
            )}
          >
            {column.cell(row, rowIndex)}
          </TableCell>
        ))}
      </TableRow>
    ));
  }

  return (
    <div className={className}>
      <Table ref={ref} aria-busy={isLoading || undefined}>
        <TableCaption className="sr-only">{caption}</TableCaption>
        <TableHeader>
          <TableRow>
            {columns.map((column) => {
              const align = column.align ?? DEFAULT_ALIGN;
              const isSortable = Boolean(column.sortValue);
              return (
                <TableHead
                  key={column.key}
                  scope="col"
                  aria-sort={
                    isSortable ? getAriaSort(activeSort, column.key) : undefined
                  }
                  className={cn(
                    "h-10",
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
                </TableHead>
              );
            })}
          </TableRow>
        </TableHeader>
        <TableBody>{renderBody()}</TableBody>
      </Table>
    </div>
  );
}
