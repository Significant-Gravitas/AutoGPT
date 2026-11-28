import type {
  MemoryFact,
  MemoryFactList,
  MemoryOverview,
  MemoryUnderstanding,
} from "../../lib/api/memory";

export interface UnderstandingRow {
  label: string;
  value: string;
}

const UNDERSTANDING_FIELDS: [keyof MemoryUnderstanding, string][] = [
  ["user_name", "Your name"],
  ["job_title", "Job title"],
  ["user_role", "Role"],
  ["business_name", "Business"],
  ["industry", "Industry"],
  ["business_size", "Size"],
  ["key_workflows", "Key workflows"],
  ["daily_activities", "Daily work"],
  ["pain_points", "Pain points"],
  ["bottlenecks", "Bottlenecks"],
  ["manual_tasks", "Manual tasks"],
  ["automation_goals", "Goals"],
  ["current_software", "Tools you use"],
  ["existing_automation", "Already automated"],
  ["additional_notes", "Notes"],
];

export function understandingRows(understanding: Partial<MemoryUnderstanding>) {
  return UNDERSTANDING_FIELDS.flatMap<UnderstandingRow>(([key, label]) => {
    const value = understandingValue(understanding[key]);
    return value ? [{ label, value }] : [];
  });
}

function understandingValue(value: MemoryUnderstanding[keyof MemoryUnderstanding] | undefined) {
  if (Array.isArray(value)) {
    return value.map((item) => item.trim()).filter(Boolean).join(", ");
  }
  return typeof value === "string" ? value.trim() : "";
}

export function factText(fact: Pick<MemoryFact, "fact" | "name" | "source" | "target">) {
  return fact.fact?.trim() || fact.name?.trim() || `${fact.source} → ${fact.target}`;
}

export function eraseWarning(overview: Pick<MemoryOverview, "facts">) {
  const noun = overview.facts === 1 ? "memory" : "memories";
  return `This erases ${overview.facts} ${noun} and what Otto knows about your work. It can't be undone.`;
}

export function withoutFact(list: MemoryFactList, uuid: string): MemoryFactList {
  return { ...list, items: list.items.filter((item) => item.uuid !== uuid) };
}

/** Puts one forgotten fact back where it was, keeping any other row forgotten since. */
export function withFactBack(
  before: MemoryFactList,
  now: MemoryFactList | undefined,
  uuid: string,
): MemoryFactList {
  if (!now) return before;
  const oldIndex = before.items.findIndex((item) => item.uuid === uuid);
  if (oldIndex < 0 || now.items.some((item) => item.uuid === uuid)) return now;
  const nextIDs = new Set(before.items.slice(oldIndex + 1).map((item) => item.uuid));
  const nextIndex = now.items.findIndex((item) => nextIDs.has(item.uuid));
  const index = nextIndex < 0 ? now.items.length : nextIndex;
  const items = now.items.slice();
  items.splice(index, 0, before.items[oldIndex]);
  return { ...now, items };
}
