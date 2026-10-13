import type { ElementNode } from "@openuidev/lang-core";

export function fieldOutputs(field: ElementNode): [string, string][] {
  const name = String(field.props.name);
  const kind =
    field.typeName === "ValidatedField"
      ? (field.props.column as { type: string }).type === "number"
        ? "number"
        : (field.props.column as { type: string }).type === "boolean"
          ? "boolean"
          : "text"
      : field.typeName;
  const outputs: [string, string][] = [[name, kind]];
  const suffixes: Record<string, [string, string][]> = {
    Comparison: [
      ["amount", "number"],
      ["label", "text"],
    ],
    CostTable: [["total", "number"]],
    DataGrid: [
      ["rows", "rows"],
      ["selected", "ids"],
    ],
    ResultList: [
      ["amount", "number"],
      ["count", "number"],
    ],
    RangeField: [
      ["low", "number"],
      ["high", "number"],
    ],
    TaskBoard: [
      ["count", "number"],
      ["done", "number"],
    ],
  };
  if (["Schedule", "Itinerary"].includes(field.typeName)) {
    for (const suffix of [
      "required_minutes",
      "capacity_minutes",
      "buffer_minutes",
      "unscheduled_minutes",
    ])
      outputs.push([`${name}_${suffix}`, "number"]);
    outputs.push([`${name}_feasible`, "boolean"]);
  }
  for (const [suffix, type] of suffixes[field.typeName] ?? [])
    outputs.push([`${name}_${suffix}`, type]);
  return outputs;
}
