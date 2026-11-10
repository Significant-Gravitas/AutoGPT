import type { MemoryUnderstanding } from "@/app/api/__generated__/models/memoryUnderstanding";

export interface UnderstandingRow {
  label: string;
  values: string[];
}

const FIELDS: [keyof MemoryUnderstanding, string][] = [
  ["user_name", "Name"],
  ["job_title", "Job title"],
  ["user_role", "Role"],
  ["business_name", "Business"],
  ["industry", "Industry"],
  ["business_size", "Team size"],
  ["key_workflows", "Key workflows"],
  ["daily_activities", "Daily work"],
  ["pain_points", "Pain points"],
  ["bottlenecks", "Bottlenecks"],
  ["manual_tasks", "Manual tasks"],
  ["automation_goals", "Goals"],
  ["current_software", "Tools you use"],
  ["existing_automation", "Existing automations"],
  ["additional_notes", "Notes"],
];

export function getUnderstandingRows(
  understanding: MemoryUnderstanding | undefined,
): UnderstandingRow[] {
  if (!understanding) return [];
  return FIELDS.flatMap(([key, label]) => {
    const values = toValues(understanding[key]);
    return values.length ? [{ label, values }] : [];
  });
}

function toValues(value: MemoryUnderstanding[keyof MemoryUnderstanding]) {
  if (Array.isArray(value)) return value.filter((item) => item.trim());
  if (typeof value === "string" && value.trim()) return [value];
  return [];
}
