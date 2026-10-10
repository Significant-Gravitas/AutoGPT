import type { ElementNode } from "@openuidev/lang-core";
import { parseFormula } from "./formula";

function nodes(value: unknown): ElementNode[] {
  if (Array.isArray(value)) return value.flatMap(nodes);
  if (!value || typeof value !== "object") return [];
  if ("typeName" in value && "props" in value) {
    const node = value as ElementNode;
    return [node, ...nodes(node.props)];
  }
  return Object.values(value).flatMap(nodes);
}

function formFields(form: ElementNode) {
  const fields = new Map<string, string>();
  for (const field of nodes(form.props.fields)) {
    const name = String(field.props.name);
    const outputs: [string, string][] = [[name, field.typeName]];
    if (field.typeName === "Comparison")
      outputs.push([`${name}_amount`, "number"], [`${name}_label`, "text"]);
    if (field.typeName === "CostTable")
      outputs.push([`${name}_total`, "number"]);
    for (const [key, kind] of outputs) {
      if (fields.has(key) || key.endsWith("__valid"))
        throw new Error(
          `Form ${form.props.name}: field ${key} collides with another field or a reserved name. Use unique names, including generated _amount, _label and _total fields.`,
        );
      fields.set(key, kind);
    }
  }
  return fields;
}

export function validateConnections(root: ElementNode) {
  const all = nodes(root);
  const forms = new Map<string, Map<string, string>>();
  for (const form of all.filter((node) => node.typeName === "Form")) {
    const name = String(form.props.name);
    if (forms.has(name)) throw new Error(`Form names must be unique: ${name}.`);
    forms.set(name, formFields(form));
  }
  for (const metric of all.filter(
    (node) => node.typeName === "CalculatedMetric",
  )) {
    const fields = forms.get(String(metric.props.form));
    if (!fields)
      throw new Error(
        `CalculatedMetric references missing Form ${metric.props.form}.`,
      );
    for (const { name, kind: referenceKind } of parseFormula(
      String(metric.props.formula),
    ).references) {
      const kind = fields.get(name);
      if (!kind)
        throw new Error(
          `CalculatedMetric: field ${name} does not exist in Form ${metric.props.form}.`,
        );
      if (
        referenceKind === "count"
          ? kind !== "MultiSelectField"
          : !["number", "NumberField"].includes(kind)
      )
        throw new Error(
          `CalculatedMetric ${metric.props.label}: use ${referenceKind === "count" ? "one MultiSelectField" : "numeric fields"}, not ${name} (${kind}).`,
        );
    }
  }
}
