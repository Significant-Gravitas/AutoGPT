import { STATUS_LABELS } from "../../../../data/helpers";
import type { TaskActivity } from "../../../../data/types";

export function describeActivity(item: TaskActivity) {
  switch (item.kind) {
    case "status_change":
      return `moved this from ${STATUS_LABELS[item.from_status]} to ${STATUS_LABELS[item.to_status]}`;
    case "created":
    case "assigned":
    case "artifact_linked":
    case "system":
      return lowerFirst(item.body);
    case "note":
      return item.body;
    default:
      return "";
  }
}

function lowerFirst(text: string) {
  if (/^[A-Z][a-z]/.test(text)) return text[0].toLowerCase() + text.slice(1);
  return text;
}
