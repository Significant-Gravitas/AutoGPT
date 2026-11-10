import type { Task } from "@/app/api/__generated__/models/task";
import { format } from "date-fns";

interface TimeRow {
  label: string;
  value: Date | null;
  formatted: string | null;
}

function toDate(value: Date | string | null | undefined) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function row(label: string, value: Date | string | null | undefined) {
  const date = toDate(value);
  return {
    label,
    value: date,
    formatted: date ? format(date, "MMM d, h:mm a") : null,
  } satisfies TimeRow;
}

/** Created always; Started and Finished show "Not yet" until they happen. */
export function timeRows(task: Task): TimeRow[] {
  return [
    row("Created", task.created_at),
    row("Started", task.started_at),
    row("Finished", task.completed_at),
  ];
}
