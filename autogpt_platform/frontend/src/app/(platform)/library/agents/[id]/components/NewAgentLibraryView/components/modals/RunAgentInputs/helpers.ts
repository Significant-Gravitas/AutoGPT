import { format, isValid, parse } from "date-fns";
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
const TIMESTAMP = /^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}/;
const MIDNIGHT_TIMESTAMP =
  /^(\d{4}-\d{2}-\d{2})[T ]00:00(?::00(?:\.0+)?)?(?:Z|[+-](?:[01]\d|2[0-3])(?::?[0-5]\d)?)?$/i;

/**
 * The day a Date input holds, as `yyyy-MM-dd`, or "" if it holds none, or
 * holds one that can't be read or doesn't exist (2026-02-30).
 *
 * Before #17305 the form stored the day picked as local midnight in UTC, so
 * 15 March picked in Auckland is `2026-03-14T11:00:00.000Z`. A timestamp at
 * midnight reads as the date written in it, as the backend reads it. Any
 * other reads as the local date at that moment, which is the day picked if it
 * was saved in this timezone.
 */
export function toCalendarDate(value: unknown): string {
  if (typeof value !== "string") return "";
  if (CALENDAR_DATE.test(value)) return existingDay(value);
  const midnight = MIDNIGHT_TIMESTAMP.exec(value);
  if (midnight) return existingDay(midnight[1]);
  const timestamp = TIMESTAMP.exec(value);
  // A day that doesn't exist would roll over into the next month.
  if (!timestamp || !existingDay(timestamp[1])) return "";
  const date = new Date(value);
  return isNaN(date.getTime()) ? "" : format(date, "yyyy-MM-dd");
}

// `day`, a yyyy-MM-dd text, if that day exists, else "".
function existingDay(day: string) {
  return isValid(parse(day, "yyyy-MM-dd", new Date())) ? day : "";
}

// The part of an object schema, such as a workflow's input schema, that names
// its fields.
interface ObjectSchema {
  properties?: Record<string, BlockIOSubSchema>;
}

/**
 * Stored inputs with each Date input's value as the day the form shows, so a
 * form seeded from a template or trigger sends that day, not the timestamp it
 * was saved as. A value the field can't show as a day becomes null: kept, it
 * would count as filled while the field shows nothing; dropped, the stored
 * value would come back when a template's inputs are merged over it. Such a
 * template or trigger therefore opens as edited, and saving it clears the
 * value. `schema` is the object schema whose properties the form shows.
 */
export function withCalendarDates(
  schema: ObjectSchema | null | undefined,
  values: Record<string, unknown> | null | undefined,
): Record<string, unknown> {
  const fields = schema?.properties ?? {};
  return Object.fromEntries(
    Object.entries(values ?? {}).flatMap(
      ([key, value]): [string, unknown][] => {
        if (!isDateField(fields, key) || isEmpty(value)) return [[key, value]];
        const day = toCalendarDate(value);
        return [[key, day || null]];
      },
    ),
  );
}

function isDateField(fields: Record<string, BlockIOSubSchema>, key: string) {
  return (
    Object.hasOwn(fields, key) &&
    resolveDataType(fields[key]).type === DataType.DATE
  );
}

function isEmpty(value: unknown) {
  return value === null || value === undefined || value === "";
}
