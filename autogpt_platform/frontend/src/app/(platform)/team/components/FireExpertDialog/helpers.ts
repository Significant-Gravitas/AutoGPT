import { ExpertDetachPreview } from "@/app/api/__generated__/models/expertDetachPreview";

export interface PauseItem {
  id: string;
  name: string;
}

export function getPauseItems(
  preview: ExpertDetachPreview | null,
): PauseItem[] {
  if (!preview) return [];
  // Type-prefixed ids keep list keys unique even when a schedule and a trigger
  // happen to share the same display name.
  return [
    ...preview.schedule_names.map((name, index) => ({
      id: `schedule-${index}`,
      name,
    })),
    ...preview.trigger_names.map((name, index) => ({
      id: `trigger-${index}`,
      name,
    })),
  ];
}

export function getAutomationLine(count: number) {
  if (count === 0) return "No automations will pause.";
  return `${count} ${count === 1 ? "automation" : "automations"} will pause.`;
}

export function getFireSummary(preview: ExpertDetachPreview | null) {
  const items = getPauseItems(preview);
  return { items, automationLine: getAutomationLine(items.length) };
}
