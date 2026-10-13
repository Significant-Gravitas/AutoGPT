import { Comparison, CostTable, CalculatedMetric } from "./catalog-connected";
import { Schedule } from "./catalog-schedule";
import { costTotal, rowTotal } from "./costs";
import { formatCalculated } from "./calculations";
import { parseFormula, evaluateFormula } from "./formula";
import { calculateSchedule, scheduleTime } from "./schedule";

export function metricText(
  props: Record<string, unknown>,
  fields: Map<string, Map<string, unknown>>,
) {
  const parsed = CalculatedMetric.props.parse(props);
  let result: number | null = null;
  try {
    result = evaluateFormula(parseFormula(parsed.formula).root, (name) =>
      fields.get(parsed.form)?.get(name),
    );
  } catch {
    /* Invalid arithmetic pauses the displayed result too. */
  }
  return [
    `${parsed.label}: ${result === null ? "Unknown / incomplete inputs" : formatCalculated(result, parsed.format, parsed.unit, parsed.precision)}`,
  ];
}

export function costText(props: Record<string, unknown>) {
  const parsed = CostTable.props.parse(props);
  const money = (amount: number | null) =>
    amount === null
      ? "Unknown"
      : formatCalculated(amount, "currency", parsed.currency, 2);
  return [
    parsed.title,
    ...parsed.rows.map(
      (row) =>
        `${row.label}: ${row.quantity ?? "unknown quantity"} × ${row.unitPrice == null ? "unknown price" : money(row.unitPrice)}${row.included ? ` = ${money(rowTotal(row))}` : " (excluded)"}`,
    ),
    `Subtotal: ${money(costTotal(parsed.rows))}`,
  ];
}

export function scheduleText(props: Record<string, unknown>) {
  const parsed = Schedule.props.parse(props);
  const result = calculateSchedule(parsed.days, parsed.tasks);
  if (!result) return [parsed.title, "Incomplete durations"];
  return [
    parsed.title,
    ...parsed.days.map(
      (day) =>
        `${day.label} available: ${day.windows.map((window) => `${window.start}–${window.end}`).join(", ")}`,
    ),
    `${result.required} min remaining; ${result.capacity} min available; ${result.feasible ? `${result.buffer} min spare` : `${result.unscheduled} min unscheduled; plan does not fit`}.`,
    ...(result.buffer < 0
      ? [`${-result.buffer} min over available time.`]
      : []),
    ...(!result.feasible ? [`${result.unused} min unused.`] : []),
    ...result.allocations.map(({ task, slots, conflict }) =>
      task.done
        ? `${task.title}: completed; excluded from remaining work`
        : `${task.title} (${task.minutes} min): ${[slots.map((slot) => `${scheduleTime(slot.start, parsed.days)}–${scheduleTime(slot.end, parsed.days)}`).join(", "), conflict].filter(Boolean).join("; ")}`,
    ),
  ];
}

export function comparisonText(props: Record<string, unknown>) {
  const parsed = Comparison.props.parse(props);
  return [
    parsed.label,
    ...parsed.options.map(
      (option) =>
        `${option.title}${parsed.value === option.id ? " (selected)" : ""}: ${option.description}\n${option.facts.map((fact) => `${fact.label}: ${fact.value}`).join("; ")}${option.amount == null ? "" : `; amount: ${option.amount}`}${option.source ? `\nSource: ${option.source.label} (${option.source.url})` : ""}`,
    ),
  ];
}
