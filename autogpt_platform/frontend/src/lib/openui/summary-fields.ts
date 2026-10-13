import { interactiveFields } from "./summary-interactive";
import type { ElementNode } from "@openuidev/lang-core";
import { Comparison, CostTable } from "./catalog-connected";
import { Schedule } from "./catalog-schedule";
import { costTotal } from "./costs";
import { calculateSchedule } from "./schedule";

export function initialFields(root: ElementNode) {
  const fields = new Map<string, Map<string, unknown>>();
  function readFields(node: unknown) {
    if (Array.isArray(node)) {
      node.forEach(readFields);
      return;
    }
    if (!isRecord(node)) return;
    if (
      node.typeName === "Form" &&
      isRecord(node.props) &&
      Array.isArray(node.props.fields)
    ) {
      const state = new Map<string, unknown>();
      fields.set(String(node.props.name), state);
      for (const field of node.props.fields) {
        if (!isRecord(field) || !isRecord(field.props)) continue;
        const props = field.props;
        const name = String(props.name);
        state.set(name, props.value);
        interactiveFields(field as unknown as ElementNode, state);
        if (field.typeName === "Comparison") {
          const parsed = Comparison.props.parse(props);
          const selected = parsed.options.find(
            (option) => option.id === parsed.value,
          );
          state.set(`${name}_amount`, selected?.amount ?? null);
        }
        if (field.typeName === "CostTable")
          state.set(
            `${name}_total`,
            costTotal(CostTable.props.parse(props).rows),
          );
        if (field.typeName === "Schedule") {
          const parsed = Schedule.props.parse(props);
          const result = calculateSchedule(parsed.days, parsed.tasks);
          for (const [suffix, value] of Object.entries({
            required_minutes: result?.required,
            capacity_minutes: result?.capacity,
            buffer_minutes: result?.buffer,
            unscheduled_minutes: result?.unscheduled,
          }))
            state.set(`${name}_${suffix}`, value ?? null);
        }
      }
    }
    Object.values(node).forEach(readFields);
  }
  readFields(root);
  return fields;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
