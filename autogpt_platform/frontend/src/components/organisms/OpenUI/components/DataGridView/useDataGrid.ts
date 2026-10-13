import { useState } from "react";
import { z } from "zod/v4";
import { DataGrid } from "@/lib/openui/catalog-data";
import {
  rowsSchema,
  compareCells,
  typedCell,
  validateRows,
  calculatedRows,
  type CellValue,
} from "@/lib/openui/data-values";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";

const draftSchema = z.object({
  rows: rowsSchema,
  selected: z.array(z.string()).max(100),
});
export function useDataGrid(props: z.infer<typeof DataGrid.props>) {
  const field = useFieldView(`${props.name}__draft`, {
    rows: props.rows,
    selected: props.selected,
  });
  const draft = draftSchema.safeParse(field.value);
  const value = draft.success
    ? draft.data
    : { rows: props.rows, selected: props.selected };
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);
  const [sort, setSort] = useState({ column: "", ascending: true });
  const errors = validateRows(props.columns, value.rows);
  const selected = value.selected.filter((id) =>
    value.rows.some((row) => row.id === id),
  );
  const current = calculatedRows(props.columns, value.rows);
  const rows = current.map((row) => ({
    ...row,
    cells: Object.fromEntries(
      props.columns.map((column) => [
        column.key,
        typedCell(row.cells[column.key] ?? null, column),
      ]),
    ),
  }));
  useDerivedField(props.name, { rows, selected });
  useDerivedField(`${props.name}_rows`, rows);
  useDerivedField(`${props.name}_selected`, selected);
  useDerivedField(`${props.name}__valid`, !errors.length);
  const filtered = current.filter((row) =>
    Object.values(row.cells).some((cell) =>
      String(cell ?? "")
        .toLowerCase()
        .includes(query.toLowerCase()),
    ),
  );
  if (sort.column)
    filtered.sort(
      (a, b) =>
        compareCells(a.cells[sort.column], b.cells[sort.column]) *
        (sort.ascending ? 1 : -1),
    );
  const currentPage = Math.min(
    page,
    Math.max(0, Math.ceil(filtered.length / 10) - 1),
  );
  function edit(id: string, key: string, cell: CellValue) {
    field.setValue({
      ...value,
      rows: value.rows.map((row) =>
        row.id === id ? { ...row, cells: { ...row.cells, [key]: cell } } : row,
      ),
    });
  }
  function select(id: string) {
    field.setValue({
      ...value,
      selected: selected.includes(id)
        ? selected.filter((item) => item !== id)
        : props.selection === "single"
          ? [id]
          : [...selected, id],
    });
  }
  function remove(id: string) {
    field.setValue({
      rows: value.rows.filter((row) => row.id !== id),
      selected: selected.filter((item) => item !== id),
    });
  }
  function add() {
    if (value.rows.length >= 100) return;
    field.setValue({
      ...value,
      rows: [
        ...value.rows,
        {
          id: `row-${crypto.randomUUID()}`,
          cells: Object.fromEntries(
            props.columns.map((column) => [column.key, null]),
          ),
        },
      ],
    });
    setQuery("");
    setSort({ column: "", ascending: true });
    setPage(Math.floor(value.rows.length / 10));
  }
  function toggleSort(column: string) {
    setSort({
      column,
      ascending: sort.column === column ? !sort.ascending : true,
    });
  }
  return {
    rows,
    selected,
    visible: filtered.slice(currentPage * 10, currentPage * 10 + 10),
    query,
    setQuery,
    page: currentPage,
    setPage,
    total: filtered.length,
    errors,
    edit,
    select,
    remove,
    add,
    toggleSort,
    sort,
  };
}
