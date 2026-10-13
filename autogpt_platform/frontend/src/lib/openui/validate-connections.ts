import type { ElementNode } from "@openuidev/lang-core";

function nodes(value: unknown): ElementNode[] {
  if (Array.isArray(value)) return value.flatMap(nodes);
  if (!value || typeof value !== "object") return [];
  if ("typeName" in value && "props" in value) {
    const node = value as ElementNode;
    return [node, ...nodes(node.props)];
  }
  return Object.values(value).flatMap(nodes);
}

function checkFieldNames(form: ElementNode) {
  const names = new Set<string>();
  for (const field of nodes(form.props.fields)) {
    for (const key of fieldOutputs(field)) {
      if (names.has(key) || key.endsWith("__valid") || key.endsWith("__draft"))
        throw new Error(
          `Form ${form.props.name}: field ${key} collides with another field or a reserved name. Use unique names, including generated _amount, _label and _count fields.`,
        );
      names.add(key);
    }
  }
}

export function validateConnections(root: ElementNode) {
  const forms = new Set<string>();
  for (const form of nodes(root).filter((node) => node.typeName === "Form")) {
    const name = String(form.props.name);
    if (forms.has(name)) throw new Error(`Form names must be unique: ${name}.`);
    forms.add(name);
    checkFieldNames(form);
  }
}

const derivedOutputs: Record<string, string[]> = {
  Comparison: ["amount", "label"],
  ResultList: ["amount", "count"],
  Itinerary: [
    "required_minutes",
    "capacity_minutes",
    "buffer_minutes",
    "unscheduled_minutes",
    "feasible",
  ],
};

function fieldOutputs(field: ElementNode) {
  const name = String(field.props.name);
  return [
    name,
    ...(derivedOutputs[field.typeName] ?? []).map(
      (suffix) => `${name}_${suffix}`,
    ),
  ];
}
