import { z } from "zod/v4";
import { asNumber } from "./calculations";

export const identifier = z.string().regex(/^[a-zA-Z][a-zA-Z0-9_-]{0,79}$/);
export const shortText = z.string().min(1).max(200);
export const sourceLink = z.object({
  label: shortText,
  url: z.url({ protocol: /^https?$/ }).max(2000),
});
export const cellValue = z
  .union([z.string().max(2000), z.number(), z.boolean()])
  .nullable();
export const dataColumn = z
  .object({
    key: identifier,
    label: shortText,
    type: z.enum([
      "text",
      "number",
      "date",
      "boolean",
      "email",
      "url",
      "select",
    ]),
    required: z.boolean().optional(),
    options: z.array(shortText).max(100).optional(),
    unit: z.string().max(30).optional(),
    calculation: z
      .object({
        operation: z.enum(["add", "subtract", "multiply", "divide"]),
        left: identifier,
        right: identifier,
      })
      .optional(),
  })
  .refine(
    (column) =>
      column.type !== "select" ||
      (!!column.options?.length &&
        new Set(column.options).size === column.options.length),
    { message: "Select columns need distinct nonempty options." },
  );
export const dataRow = z.object({
  id: identifier,
  cells: z.record(identifier, cellValue),
  source: sourceLink.optional(),
});
export const columnsSchema = z
  .array(dataColumn)
  .min(1)
  .max(8)
  .refine(
    (columns) =>
      new Set(columns.map((column) => column.key)).size === columns.length,
    { message: "Column keys must be unique." },
  )
  .refine(
    (columns) =>
      columns.every(
        (column) =>
          !column.calculation ||
          (column.type === "number" &&
            [column.calculation.left, column.calculation.right].every((key) =>
              columns.some(
                (input) =>
                  input.key === key &&
                  input.type === "number" &&
                  !input.calculation,
              ),
            )),
      ),
    {
      message:
        "Calculated numeric columns must reference two existing, non-calculated numeric columns.",
    },
  );
export const rowsSchema = z
  .array(dataRow)
  .max(100)
  .refine((rows) => new Set(rows.map((row) => row.id)).size === rows.length, {
    message: "Row ids must be unique.",
  });
export type DataColumn = z.infer<typeof dataColumn>;
export type DataRow = z.infer<typeof dataRow>;
export type CellValue = z.infer<typeof cellValue>;

export function cellError(value: unknown, column: DataColumn) {
  if (value === null || value === undefined || value === "")
    return column.required ? "Fill in this field." : "";
  if (column.type === "text" && typeof value !== "string") return "Enter text.";
  if (column.type === "number" && asNumber(value) === null)
    return "Enter a number, such as 10 or 10.5.";
  if (column.type === "boolean" && typeof value !== "boolean")
    return "Choose yes or no.";
  if (column.type === "date" && !z.iso.date().safeParse(value).success)
    return "Choose a valid date.";
  if (column.type === "email" && !z.email().safeParse(value).success)
    return "Enter a valid email address.";
  if (
    column.type === "url" &&
    !z.url({ protocol: /^https?$/ }).safeParse(value).success
  )
    return "Enter an http or https URL.";
  if (column.type === "select" && !column.options?.includes(String(value)))
    return "Choose an available option.";
  return "";
}

export function typedCell(value: CellValue, column: DataColumn): CellValue {
  if (value === "" || value == null) return null;
  if (column.type === "number") return asNumber(value);
  return value;
}

export function compareCells(a: unknown, b: unknown) {
  if (a == null || a === "") return b == null || b === "" ? 0 : 1;
  if (b == null || b === "") return -1;
  const left = asNumber(a);
  const right = asNumber(b);
  if (left !== null && right !== null) return left - right;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function validateRows(columns: DataColumn[], rows: DataRow[]) {
  return calculatedRows(columns, rows).flatMap((row) => [
    ...Object.keys(row.cells)
      .filter((key) => !columns.some((column) => column.key === key))
      .map((key) => `Unknown column ${key}.`),
    ...columns.flatMap((column) => {
      const error = cellError(row.cells[column.key], column);
      return error ? [`${row.id} / ${column.label}: ${error}`] : [];
    }),
  ]);
}

export function calculatedRows(columns: DataColumn[], rows: DataRow[]) {
  return rows.map((row) => {
    const cells = { ...row.cells };
    for (const column of columns) {
      if (!column.calculation) continue;
      const { operation, left, right } = column.calculation;
      const a = asNumber(cells[left]),
        b = asNumber(cells[right]);
      const value =
        a === null || b === null
          ? null
          : operation === "add"
            ? a + b
            : operation === "subtract"
              ? a - b
              : operation === "multiply"
                ? a * b
                : b === 0
                  ? null
                  : a / b;
      cells[column.key] =
        value !== null && Number.isFinite(value) ? value : null;
    }
    return { ...row, cells };
  });
}
