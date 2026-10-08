import { useState } from "react";

export function useDataTableView(rows: string[][]) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState({ column: -1, ascending: true });
  const visibleRows = rows.slice(0, 30).filter((row) =>
    row.some((cell) =>
      String(cell ?? "")
        .toLowerCase()
        .includes(query.toLowerCase()),
    ),
  );
  if (sort.column >= 0)
    visibleRows.sort(
      (a, b) =>
        String(a[sort.column] ?? "").localeCompare(
          String(b[sort.column] ?? ""),
          undefined,
          { numeric: true },
        ) * (sort.ascending ? 1 : -1),
    );
  function toggleSort(column: number) {
    setSort((current) => ({
      column,
      ascending: current.column === column ? !current.ascending : true,
    }));
  }
  return { query, setQuery, sort, toggleSort, visibleRows };
}
