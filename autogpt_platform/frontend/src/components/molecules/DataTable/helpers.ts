import type { ReactNode } from "react";

export type SortDirection = "asc" | "desc";

export interface DataTableSort {
  key: string;
  direction: SortDirection;
}

type SortValue = string | number | boolean | Date | null | undefined;

export interface DataTableColumn<T> {
  key: string;
  header: ReactNode;
  cell: (row: T, index: number) => ReactNode;
  /** Value to sort by. A column is sortable when this is set. */
  sortValue?: (row: T) => SortValue;
  align?: "left" | "center" | "right";
  className?: string;
  headerClassName?: string;
}

export const DEFAULT_ALIGN = "left" as const;

export const alignClassName = {
  left: "text-left",
  center: "text-center",
  right: "text-right",
} as const;

function compareValues(a: SortValue, b: SortValue): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (a instanceof Date && b instanceof Date) return a.getTime() - b.getTime();
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "boolean" && typeof b === "boolean")
    return Number(a) - Number(b);
  return String(a).localeCompare(String(b), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

/** Sorts a copy of `rows`. Empty values always sort last. */
export function sortRows<T>(
  rows: T[],
  columns: DataTableColumn<T>[],
  sort: DataTableSort | null,
): T[] {
  if (!sort) return rows;
  const column = columns.find((candidate) => candidate.key === sort.key);
  const getValue = column?.sortValue;
  if (!getValue) return rows;

  return [...rows].sort((a, b) => {
    const left = getValue(a);
    const right = getValue(b);
    if (left == null || right == null) return compareValues(left, right);
    const result = compareValues(left, right);
    return sort.direction === "asc" ? result : -result;
  });
}

/** Ascending, then descending, then unsorted. */
export function getNextSort(
  current: DataTableSort | null,
  key: string,
): DataTableSort | null {
  if (current?.key !== key) return { key, direction: "asc" };
  if (current.direction === "asc") return { key, direction: "desc" };
  return null;
}

export function getAriaSort(sort: DataTableSort | null, key: string) {
  if (sort?.key !== key) return "none" as const;
  return sort.direction === "asc"
    ? ("ascending" as const)
    : ("descending" as const);
}
