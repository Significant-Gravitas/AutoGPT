import { format } from "date-fns";
import {
  type BlockIOSubSchema,
  DataType,
  resolveDataType,
} from "@/lib/autogpt-server-api/types";

export function getSelectOptions(schema: BlockIOSubSchema) {
  if (!("enum" in schema) || !Array.isArray(schema.enum)) return [];
  return schema.enum.filter((option) => option);
}

const CALENDAR_DATE = /^\d{4}-\d{2}-\d{2}$/;
const TIMESTAMP = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/;
const MIDNIGHT_TIMESTAMP =
  /^(\d{4}-\d{2}-\d{2})[T ]00:00(?::00(?:\.0+)?)?(?:Z|[+-]\d{2}:?\d{2})?$/i;

/**
 * The day a Date input holds, as `yyyy-MM-dd`, or "" if it holds none.
 *
 * Before #17305 the form stored the day picked as local midnight in UTC, so
 * 15 March picked in Auckland is `2026-03-14T11:00:00.000Z`. A timestamp at
 * midnight reads as the date written in it, as the backend reads it. Any
 * other reads as the local date at that moment, which is the day picked if it
 * was saved in this timezone.
 */
export function toCalendarDate(value: unknown): string {
  if (typeof value !== "string") return "";
  if (CALENDAR_DATE.test(value)) return value;
  const midnight = MIDNIGHT_TIMESTAMP.exec(value);
  if (midnight) return midnight[1];
  if (!TIMESTAMP.test(value)) return "";
  const date = new Date(value);
  return isNaN(date.getTime()) ? "" : format(date, "yyyy-MM-dd");
}

// The part of an object schema, such as a workflow's input schema, that names
// its fields.
interface ObjectSchema {
  properties?: Record<string, BlockIOSubSchema>;
}

/**
 * Stored inputs with each Date input's value as the day the form shows, so a
 * form seeded from a template or trigger sends that day, not the timestamp it
 * was saved as. `schema` is the object schema whose properties the form shows.
 */
export function withCalendarDates(
  schema: ObjectSchema | null | undefined,
  values: Record<string, unknown> | null | undefined,
): Record<string, unknown> {
  const fields = schema?.properties ?? {};
  return Object.fromEntries(
    Object.entries(values ?? {}).map(([key, value]) => {
      if (!isDateField(fields, key)) return [key, value];
      return [key, toCalendarDate(value) || value];
    }),
  );
}

function isDateField(fields: Record<string, BlockIOSubSchema>, key: string) {
  return (
    Object.hasOwn(fields, key) &&
    resolveDataType(fields[key]).type === DataType.DATE
  );
}
