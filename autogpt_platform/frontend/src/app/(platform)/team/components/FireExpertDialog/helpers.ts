import { ExpertDetachPreview } from "@/app/api/__generated__/models/expertDetachPreview";

export function getPauseNames(preview: ExpertDetachPreview | null) {
  if (!preview) return [];
  return [...preview.schedule_names, ...preview.trigger_names];
}

export function getScheduleLine(count: number) {
  if (count === 0) return "No scheduled runs will pause.";
  return `${count} scheduled ${count === 1 ? "run" : "runs"} will pause.`;
}

export function getFireSummary(preview: ExpertDetachPreview | null) {
  const names = getPauseNames(preview);
  return { names, scheduleLine: getScheduleLine(names.length) };
}
