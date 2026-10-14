import { SeriesChart } from "./catalog-sections";
import { Artifact } from "./catalog-artifacts";
import { Itinerary } from "./catalog-itinerary";
import { Comparison, ResultList } from "./catalog-results";
import { Prose } from "./catalog";
import { planItinerary } from "./itinerary";
import { scheduleTime } from "./schedule";

export function sectionText(
  type: string,
  props: Record<string, unknown>,
): string[] | null {
  if (type === "Prose") return [Prose.props.parse(props).markdown];
  if (type === "Artifact") {
    const p = Artifact.props.parse(props);
    return [p.title, p.description, `${p.filename} (${p.mimeType}): ${p.uri}`];
  }
  if (type === "SeriesChart") {
    const p = SeriesChart.props.parse(props);
    return [
      p.title,
      p.description,
      ...p.points.map(
        (point) =>
          `${point.label}: ${p.series.map((series) => `${series.label} ${point.values[series.key] ?? "Unknown"} ${p.unit}`).join("; ")}`,
      ),
    ];
  }
  if (type === "Itinerary") {
    const p = Itinerary.props.parse(props);
    const { result, error } = planItinerary(p, {
      days: p.days,
      tasks: p.tasks,
      timeZone: p.timeZone,
      selected: p.locations[0].id,
    });
    return [
      p.title,
      `Proposed plan; ${p.timeZone}`,
      ...p.days.map(
        (day) =>
          `${day.date}: ${day.windows.map((window) => `${window.start}–${window.end}`).join(", ")}`,
      ),
      ...(result
        ? [
            `${result.required} min remaining; ${result.capacity} min available; ${result.feasible ? `${result.buffer} min spare` : `${result.unscheduled} min unscheduled`}`,
            ...result.allocations.map(({ task, slots, conflict }) =>
              task.done
                ? `${task.title}: completed`
                : `${task.title}: ${slots.map((slot) => `${scheduleTime(slot.start, p.days)}–${scheduleTime(slot.end, p.days)}`).join(", ")} ${conflict}`,
            ),
          ]
        : [error]),
      ...p.locations.map((place) => `${place.name}: ${place.detail}`),
    ];
  }
  if (type === "ResultList") {
    const p = ResultList.props.parse(props);
    return [
      p.label,
      ...p.options.map(
        (option) =>
          `${p.value.includes(option.id) ? "[selected] " : ""}${option.title}: ${option.description}\n${option.facts.map((fact) => `${fact.label}: ${fact.value}${fact.source ? ` (${fact.source.url})` : ""}`).join("; ")}${option.amount == null ? "" : `; ${option.amount} ${p.unit}`}${option.source ? `\nSource: ${option.source.url}` : ""}`,
      ),
    ];
  }
  if (type === "Comparison") {
    const p = Comparison.props.parse(props);
    return [
      p.label,
      ...p.options.map(
        (option) =>
          `${option.title}${p.value === option.id ? " (selected)" : ""}: ${option.description}\n${option.facts.map((fact) => `${fact.label}: ${fact.value}`).join("; ")}${option.amount == null ? "" : `; amount: ${option.amount}`}${option.source ? `\nSource: ${option.source.label} (${option.source.url})` : ""}`,
      ),
    ];
  }
  return null;
}
