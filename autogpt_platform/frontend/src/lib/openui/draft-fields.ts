import type { ElementNode } from "@openuidev/lang-core";
import { parseWorkspace } from "./parse";

export interface DraftField {
  form: string | null;
  key: string;
  value: unknown;
  contract: unknown;
  props: Record<string, unknown>;
  type: string;
}
const privateDrafts = new Set([
  "DataGrid",
  "FieldGroup",
  "ValidatedField",
  "RangeField",
  "DateTimeField",
  "TaskBoard",
  "Itinerary",
]);
export function draftFields(source: string) {
  const result: DraftField[] = [];
  function visit(node: ElementNode) {
    const props = node.props;
    if (node.typeName === "Form")
      for (const field of props.fields as ElementNode[]) {
        const p = field.props;
        let value = p.value;
        if (["CostTable", "FieldGroup"].includes(field.typeName))
          value = p.rows;
        if (["Schedule", "TaskBoard"].includes(field.typeName)) value = p.tasks;
        if (field.typeName === "DataGrid")
          value = { rows: p.rows, selected: p.selected };
        if (field.typeName === "Itinerary")
          value = {
            days: p.days,
            tasks: p.tasks,
            timeZone: p.timeZone,
            selected: (p.locations as { id: string }[])[0].id,
          };
        result.push({
          form: String(props.name),
          key: `${p.name}${privateDrafts.has(field.typeName) ? "__draft" : ""}`,
          value,
          type: field.typeName,
          props: p,
          contract: [
            field.typeName,
            p.columns ?? p.column ?? null,
            p.selection ?? p.multiple ?? null,
          ],
        });
      }
    if (node.typeName === "Checklist")
      result.push({
        form: null,
        key: `checklist:${node.statementId ?? props.title}`,
        type: "Checklist",
        props,
        value: (props.items as { done?: boolean }[]).flatMap((item, index) =>
          item.done ? [index] : [],
        ),
        contract: ["Checklist"],
      });
    if (node.typeName === "Dashboard")
      result.push({
        form: null,
        key: `dashboard:${props.name}`,
        type: "Dashboard",
        props,
        value: { query: "", categories: {}, ranges: {}, selectedOnly: false },
        contract: ["Dashboard", props.columns, props.binding ?? null],
      });
    for (const child of (props.children as ElementNode[] | undefined) ?? [])
      visit(child);
  }
  visit(parseWorkspace(source));
  return result;
}

export function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
export function fieldValue(state: Record<string, unknown>, field: DraftField) {
  const value = field.form
    ? record(state[field.form])?.[field.key]
    : state[field.key];
  return record(value) && "componentType" in record(value)!
    ? record(value)!.value
    : value;
}
export function packFields(fields: { field: DraftField; value: unknown }[]) {
  const state: Record<string, unknown> = Object.create(null);
  for (const { field, value } of fields) {
    const wrapped = { value, componentType: "Field" };
    if (field.form)
      state[field.form] = {
        ...record(state[field.form]),
        [field.key]: wrapped,
      };
    else state[field.key] = wrapped;
  }
  return state;
}
export function persistedState(source: string, state: Record<string, unknown>) {
  return packFields(
    draftFields(source).flatMap((field) => {
      const value = fieldValue(state, field);
      return value === undefined ? [] : [{ field, value }];
    }),
  );
}
