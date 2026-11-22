import type { CollectionRowRecord } from "@/app/api/__generated__/models/collectionRowRecord";
import type { Condition } from "@/app/api/__generated__/models/condition";
import type { FieldType } from "@/app/api/__generated__/models/fieldType";
import type { PageTableColumn } from "@/app/api/__generated__/models/pageTableColumn";

export type TableColumn = PageTableColumn;
export type TableRow = CollectionRowRecord;

/** A cell as text, the way the table shows it when it is not being edited. */
export function cellText(column: TableColumn, value: unknown): string {
  if (value === null || value === undefined || value === "") return "";
  if (column.type === "checkbox") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(String).join(", ");
  if (column.type === "number" && typeof value === "number") {
    return withUnit(
      Number.isInteger(value)
        ? value.toLocaleString("en-US")
        : value.toLocaleString("en-US", { maximumFractionDigits: 2 }),
      column.unit,
    );
  }
  if (column.type === "url" && typeof value === "string") return hostOf(value);
  return String(value);
}

/** What a person typed, as the value the store converts. */
export function parseInput(type: FieldType, text: string): unknown {
  const trimmed = text.trim();
  if (!trimmed) return null;
  if (type === "multi_select") {
    return trimmed
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean);
  }
  return trimmed;
}

/** The text an edit box starts with. */
export function editText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.map(String).join(", ");
  return String(value);
}

export function inputType(type: FieldType) {
  if (type === "number") return "number" as const;
  if (type === "date") return "date" as const;
  if (type === "url") return "url" as const;
  if (type === "email") return "email" as const;
  return "text" as const;
}

/** The section's stored filter plus the chips a person flipped, as the rows route takes it. */
export function whereParam(
  filter: Condition[],
  chipField: string | null,
  chips: string[],
): string | undefined {
  const withChips = chipField && chips.length ? { [chipField]: chips } : {};
  if (!filter.length && !Object.keys(withChips).length) return undefined;
  return JSON.stringify({ conditions: filter, chips: withChips });
}

/** The first select column: its options are the chips above the table. */
export function chipColumn(columns: TableColumn[]): TableColumn | null {
  return columns.find((c) => c.type === "select" && c.options?.length) ?? null;
}

export function isNumeric(column: TableColumn) {
  return column.type === "number";
}

function withUnit(text: string, unit: string | null | undefined) {
  if (!unit) return text;
  return unit.length === 1 && !/[a-z0-9]/i.test(unit)
    ? `${unit}${text}`
    : `${text} ${unit}`;
}

function hostOf(url: string) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
