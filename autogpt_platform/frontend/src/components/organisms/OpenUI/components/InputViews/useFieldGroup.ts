import { useGetFieldValue } from "@openuidev/react-lang";
import { useFormName } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { FieldGroup } from "@/lib/openui/catalog-inputs";
import {
  rowsSchema,
  typedCell,
  validateRows,
  type CellValue,
} from "@/lib/openui/data-values";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";

export function useFieldGroup(props: z.infer<typeof FieldGroup.props>) {
  const form = useFormName();
  const get = useGetFieldValue();
  const field = useFieldView(`${props.name}__draft`, props.rows);
  const parsed = rowsSchema.safeParse(field.value);
  const rows = parsed.success ? parsed.data : props.rows;
  const visible =
    !props.visibleWhen ||
    get(form, props.visibleWhen.field) === props.visibleWhen.equals;
  const error = visible
    ? props.required && !rows.length
      ? "Add at least one entry."
      : (validateRows(props.columns, rows)[0] ?? "")
    : "";
  useDerivedField(
    props.name,
    visible
      ? rows.map((row) => ({
          id: row.id,
          cells: Object.fromEntries(
            props.columns.map((column) => [
              column.key,
              typedCell(row.cells[column.key] ?? null, column),
            ]),
          ),
        }))
      : [],
  );
  useDerivedField(`${props.name}__valid`, !error);
  function edit(id: string, key: string, value: CellValue) {
    field.setValue(
      rows.map((row) =>
        row.id === id ? { ...row, cells: { ...row.cells, [key]: value } } : row,
      ),
    );
  }
  function remove(id: string) {
    field.setValue(rows.filter((row) => row.id !== id));
  }
  function add() {
    if (rows.length < 20)
      field.setValue([
        ...rows,
        {
          id: `entry-${crypto.randomUUID()}`,
          cells: Object.fromEntries(
            props.columns.map((column) => [
              column.key,
              column.type === "boolean" ? false : null,
            ]),
          ),
        },
      ]);
  }
  return { rows, visible, error, edit, remove, add };
}
