import type { ElementNode } from "@openuidev/lang-core";
import { DataGrid } from "./catalog-data";
import { ResultList, RankedList } from "./catalog-results";
import {
  DateRangeField,
  FieldGroup,
  RangeField,
  SearchSelect,
  ValidatedField,
  DocumentEditor,
} from "./catalog-inputs";
import { TaskBoard } from "./catalog-tasks";
import { DateTimeField } from "./catalog-temporal";
import { resolveDateTime } from "./temporal";
import {
  cellError,
  calculatedRows,
  typedCell,
  type DataColumn,
  type DataRow,
} from "./data-values";
import { Itinerary } from "./catalog-itinerary";
import { planItinerary } from "./itinerary";

export function interactiveFields(
  node: ElementNode,
  fields: Map<string, unknown>,
) {
  const { typeName: type, props } = node;
  const name = String(props.name);
  function set(suffix: string, value: unknown) {
    fields.set(`${name}${suffix}`, value);
  }
  if (type === "DataGrid") {
    const p = DataGrid.props.parse(props);
    const rows = calculatedRows(p.columns, p.rows);
    set("", { rows, selected: p.selected });
    set("_rows", rows);
    set("_selected", p.selected);
  }
  if (type === "TaskBoard") {
    const p = TaskBoard.props.parse(props);
    set("", p.tasks);
    set("_count", p.tasks.length);
    set("_done", p.tasks.filter((task) => task.status === "done").length);
  }
  if (type === "ResultList") {
    const p = ResultList.props.parse(props);
    const selected = p.options.filter((option) => p.value.includes(option.id));
    set("_count", selected.length);
    set(
      "_amount",
      selected.length && selected.every((option) => option.amount != null)
        ? selected.reduce((sum, option) => sum + option.amount!, 0)
        : null,
    );
  }
  if (type === "ValidatedField") {
    const p = ValidatedField.props.parse(props);
    set(
      "",
      cellError(p.value, p.column)
        ? null
        : typedCell(p.value ?? null, p.column),
    );
  }
  if (type === "RangeField") {
    const p = RangeField.props.parse(props);
    set("_low", p.value.low);
    set("_high", p.value.high);
  }
  if (type === "DateTimeField") {
    const p = DateTimeField.props.parse(props);
    set("", {
      ...p.value,
      instant: resolveDateTime(p.value, p.required).instant,
    });
  }
  if (type === "Itinerary") {
    const p = Itinerary.props.parse(props);
    const plan = planItinerary(p, {
      days: p.days,
      tasks: p.tasks,
      timeZone: p.timeZone,
      selected: p.locations[0].id,
    });
    for (const [suffix, value] of Object.entries({
      required_minutes: plan.result?.required,
      capacity_minutes: plan.result?.capacity,
      buffer_minutes: plan.result?.buffer,
      unscheduled_minutes: plan.result?.unscheduled,
      feasible: plan.result?.feasible,
    }))
      set(`_${suffix}`, value ?? null);
  }
}

export function tableText(columns: DataColumn[], rows: DataRow[]) {
  return [
    columns.map((column) => column.label).join(" | "),
    ...rows.map((row) =>
      columns.map((column) => row.cells[column.key] ?? "Unknown").join(" | "),
    ),
  ].join("\n");
}

export function interactiveText(
  type: string,
  props: Record<string, unknown>,
  fields: Map<string, unknown> | undefined,
): string[] | null {
  if (type === "DataGrid") {
    const p = DataGrid.props.parse(props);
    return [
      p.title,
      tableText(p.columns, calculatedRows(p.columns, p.rows)),
      `Selected: ${p.selected.join(", ") || "none"}`,
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
  if (type === "RankedList") {
    const p = RankedList.props.parse(props);
    return [
      p.label,
      ...p.value.map(
        (id, index) =>
          `${index + 1}. ${p.items.find((item) => item.id === id)?.label}`,
      ),
    ];
  }
  if (type === "SearchSelect") {
    const p = SearchSelect.props.parse(props);
    return [
      `${p.label}: ${p.value.map((id) => p.options.find((option) => option.id === id)?.label).join(", ") || "Not selected"}`,
    ];
  }
  if (type === "ValidatedField") {
    const p = ValidatedField.props.parse(props);
    return [
      `${p.column.label}: ${cellError(p.value, p.column) ? "Needs correction" : (p.value ?? "Unknown")}`,
    ];
  }
  if (type === "FieldGroup") {
    const p = FieldGroup.props.parse(props);
    return p.visibleWhen &&
      fields?.get(p.visibleWhen.field) !== p.visibleWhen.equals
      ? []
      : [p.label, tableText(p.columns, p.rows)];
  }
  if (type === "DateRangeField") {
    const p = DateRangeField.props.parse(props);
    return [
      `${p.label}: ${p.value.start || "Unknown"} – ${p.value.end || "Unknown"}`,
    ];
  }
  if (type === "RangeField") {
    const p = RangeField.props.parse(props);
    return [
      `${p.label}: ${p.value.low ?? "Unknown"} – ${p.value.high ?? "Unknown"} ${p.unit}`,
    ];
  }
  if (type === "DateTimeField") {
    const p = DateTimeField.props.parse(props);
    const resolved = resolveDateTime(p.value, p.required);
    return [
      `${p.label}: ${p.value.local || "Unknown"} ${p.value.timeZone}${resolved.error ? `; ${resolved.error}` : resolved.instant ? ` (${resolved.instant})` : ""}`,
    ];
  }
  if (type === "DocumentEditor") {
    const p = DocumentEditor.props.parse(props);
    return [p.title, p.value];
  }
  if (type === "TaskBoard") {
    const p = TaskBoard.props.parse(props);
    return [
      p.title,
      "User-reported progress",
      ...p.tasks.map(
        (task) =>
          `${task.title}: ${task.status}; owner ${task.owner || "unassigned"}; due ${task.due || "unset"}${task.dependsOn.length ? `; requires ${task.dependsOn.map((id) => p.tasks.find((item) => item.id === id)?.title).join(", ")}` : ""}`,
      ),
    ];
  }
  return null;
}
