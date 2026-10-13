import { Dashboard, SeriesChart } from "./catalog-dashboard";
import { Artifact, ExecutionView } from "./catalog-artifacts";
import { Itinerary } from "./catalog-itinerary";
import { planItinerary } from "./itinerary";
import { aggregateRows } from "./dashboard";
import { calculatedRows, rowsSchema } from "./data-values";
import { tableText } from "./summary-interactive";
import { scheduleTime } from "./schedule";

export function sectionText(
  type: string,
  props: Record<string, unknown>,
  fields: Map<string, Map<string, unknown>>,
): string[] | null {
  if (type === "Artifact") {
    const p = Artifact.props.parse(props);
    return [p.title, p.description, `${p.filename} (${p.mimeType}): ${p.uri}`];
  }
  if (type === "ExecutionView") {
    const p = ExecutionView.props.parse(props);
    return [
      p.title,
      `Execution ${p.executionId}. Open the live view for current backend status and outputs.`,
    ];
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
  if (type === "Dashboard") {
    const p = Dashboard.props.parse(props);
    const rows = p.binding
      ? rowsSchema.parse(
          fields.get(p.binding.form)?.get(`${p.binding.field}_rows`),
        )
      : calculatedRows(p.columns, p.rows);
    return [
      p.title,
      "Snapshot before interactive filtering",
      ...p.metrics.map(
        (metric) =>
          `${metric.label}: ${aggregateRows(rows, metric) ?? "Unknown"} ${metric.unit}`,
      ),
      tableText(p.columns, rows),
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
  return null;
}
