import type { RJSFSchema } from "@rjsf/utils";
import type { CustomEdge } from "../edges/CustomEdge";

export type OutputProperties = Record<string, RJSFSchema | boolean>;

function isSchemaObject(value: unknown): value is RJSFSchema {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function getOutputProperties(schema: unknown): OutputProperties {
  return isSchemaObject(schema) &&
    Object.hasOwn(schema, "properties") &&
    isSchemaObject(schema.properties)
    ? schema.properties
    : {};
}

function withOutputPath(
  schema: unknown,
  segments: string[],
  index: number,
): RJSFSchema | undefined {
  if (schema !== true && !isSchemaObject(schema)) return;
  const field = schema === true ? {} : schema;
  if (
    field.type !== undefined &&
    field.type !== "object" &&
    !(Array.isArray(field.type) && field.type.includes("object"))
  )
    return;

  const properties = getOutputProperties(field);
  const key = segments[index];
  const exists = Object.hasOwn(properties, key);
  if (!exists && field.additionalProperties === false) return;
  if (
    !field.type &&
    !Object.keys(properties).length &&
    Array.isArray(field.anyOf)
  ) {
    for (const option of field.anyOf) {
      const expanded = withOutputPath(option, segments, index);
      if (expanded)
        return {
          ...expanded,
          ...field,
          type: expanded.type,
          properties: expanded.properties,
        };
    }
    return;
  }

  let child = exists ? properties[key] : (field.additionalProperties ?? true);
  if (!exists) {
    if (child !== true && !isSchemaObject(child)) return;
    child = { title: key, ...(child === true ? {} : child) };
  }
  if (index < segments.length - 1) {
    const expanded = withOutputPath(child, segments, index + 1);
    if (!expanded) return;
    child = expanded;
  }

  return {
    ...field,
    type: field.type ?? "object",
    properties: { ...properties, [key]: child },
  };
}

export function augmentOutputSchemaWithEdges(
  outputSchema: RJSFSchema | undefined,
  nodeID: string,
  edges: readonly Pick<CustomEdge, "source" | "sourceHandle">[],
): OutputProperties {
  const properties = { ...getOutputProperties(outputSchema) };
  for (const edge of edges) {
    if (edge?.source !== nodeID || typeof edge.sourceHandle !== "string")
      continue;
    const segments = edge.sourceHandle.split("_#_");
    if (
      segments.length < 2 ||
      segments.length > 64 ||
      segments.some((segment) => !segment) ||
      !Object.hasOwn(properties, segments[0])
    )
      continue;

    const expanded = withOutputPath(properties[segments[0]], segments, 1);
    if (expanded) {
      Object.defineProperty(properties, segments[0], {
        value: expanded,
        enumerable: true,
        configurable: true,
        writable: true,
      });
    }
  }
  return properties;
}
