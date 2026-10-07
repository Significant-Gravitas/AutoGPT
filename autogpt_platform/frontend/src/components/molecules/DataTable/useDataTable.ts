import { useState } from "react";
import {
  DataTableColumn,
  DataTableSort,
  getNextSort,
  sortRows,
} from "./helpers";

interface Args<T> {
  rows: T[];
  columns: DataTableColumn<T>[];
  sort?: DataTableSort | null;
  defaultSort?: DataTableSort | null;
  onSortChange?: (sort: DataTableSort | null) => void;
  manualSorting: boolean;
}

export function useDataTable<T>({
  rows,
  columns,
  sort,
  defaultSort = null,
  onSortChange,
  manualSorting,
}: Args<T>) {
  const [internalSort, setInternalSort] = useState<DataTableSort | null>(
    defaultSort,
  );
  const activeSort = sort === undefined ? internalSort : sort;

  function handleSort(key: string) {
    const next = getNextSort(activeSort, key);
    if (sort === undefined) setInternalSort(next);
    onSortChange?.(next);
  }

  const visibleRows = manualSorting
    ? rows
    : sortRows(rows, columns, activeSort);

  return { activeSort, visibleRows, handleSort };
}
