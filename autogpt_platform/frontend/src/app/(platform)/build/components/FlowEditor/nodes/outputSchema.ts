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

const MAX_OUTPUT_SCHEMA_DEPTH = 6;
const COMBINATORS = ["anyOf", "oneOf", "allOf"] as const;

function combinatorBranches(schema: RJSFSchema): (RJSFSchema | boolean)[] {
  return COMBINATORS.flatMap((keyword) =>
    Array.isArray(schema[keyword]) ? schema[keyword] : [],
  );
}

function isObjectOutputSchema(schema: RJSFSchema): boolean {
  if (schema.type === "object") return true;
  if (Array.isArray(schema.type) && schema.type.includes("object")) return true;
  return Object.keys(getOutputProperties(schema)).length > 0;
}

function copyOutputProperties(source: OutputProperties): OutputProperties {
  const copy: OutputProperties = {};
  for (const [key, value] of Object.entries(source)) {
    Object.defineProperty(copy, key, {
      value,
      enumerable: true,
      configurable: true,
      writable: true,
    });
  }
  return copy;
}

function mergeObjectProperties(
  current: RJSFSchema | boolean,
  incoming: RJSFSchema | boolean,
  depth: number,
): RJSFSchema | undefined {
  if (
    depth >= MAX_OUTPUT_SCHEMA_DEPTH ||
    !isSchemaObject(current) ||
    !isSchemaObject(incoming) ||
    !isObjectOutputSchema(current) ||
    !isObjectOutputSchema(incoming)
  )
    return undefined;

  const incomingProperties = getNestedOutputProperties(incoming);
  if (Object.keys(incomingProperties).length === 0) return undefined;
  const merged = copyOutputProperties(getNestedOutputProperties(current));
  let grew = false;
  for (const [childKey, childValue] of Object.entries(incomingProperties)) {
    const before = Object.getOwnPropertyDescriptor(merged, childKey)?.value;
    assignOwnProperty(merged, childKey, childValue, depth + 1);
    if (Object.getOwnPropertyDescriptor(merged, childKey)?.value !== before)
      grew = true;
  }
  if (!grew) return undefined;
  return { ...current, properties: merged };
}

function assignOwnProperty(
  target: OutputProperties,
  key: string,
  value: RJSFSchema | boolean,
  depth = 0,
) {
  const current = Object.getOwnPropertyDescriptor(target, key)?.value as
    | RJSFSchema
    | boolean
    | undefined;
  const next =
    current === undefined
      ? value
      : (mergeObjectProperties(current, value, depth) ?? current);
  if (next === current) return;
  Object.defineProperty(target, key, {
    value: next,
    enumerable: true,
    configurable: true,
    writable: true,
  });
}

function nestedOutputProperties(
  schema: unknown,
  depth: number,
  seen: Set<object>,
): OutputProperties {
  const direct = getOutputProperties(schema);
  if (
    !isSchemaObject(schema) ||
    depth >= MAX_OUTPUT_SCHEMA_DEPTH ||
    seen.has(schema)
  )
    return direct;

  const hasDirect = Object.keys(direct).length > 0;
  const branches = [
    schema.allOf,
    ...(hasDirect ? [] : [schema.anyOf, schema.oneOf]),
  ].filter((branch): branch is (RJSFSchema | boolean)[] =>
    Array.isArray(branch),
  );
  if (branches.length === 0) return direct;

  seen.add(schema);
  const merged = copyOutputProperties(direct);
  for (const branch of branches) {
    for (const option of branch) {
      const nested = nestedOutputProperties(option, depth + 1, seen);
      for (const [key, value] of Object.entries(nested)) {
        assignOwnProperty(merged, key, value);
      }
    }
  }
  return merged;
}

export function getNestedOutputProperties(schema: unknown): OutputProperties {
  return nestedOutputProperties(schema, 0, new Set());
}

function declaresOutputKey(
  schema: unknown,
  key: string,
  seen: Set<object>,
): boolean {
  if (!isSchemaObject(schema) || seen.has(schema)) return false;
  const properties = getOutputProperties(schema);
  if (Object.hasOwn(properties, key)) return true;
  if (Object.keys(properties).length > 0) return false;
  seen.add(schema);
  return combinatorBranches(schema).some((option) =>
    declaresOutputKey(option, key, seen),
  );
}

function expandCombinator(
  field: RJSFSchema,
  segments: string[],
  index: number,
): RJSFSchema | undefined {
  const branches = combinatorBranches(field);
  const declared = branches.filter((option) =>
    declaresOutputKey(option, segments[index], new Set()),
  );
  for (const option of declared.length > 0 ? declared : branches) {
    const expanded = withOutputPath(option, segments, index);
    if (!expanded) continue;
    return {
      ...expanded,
      ...field,
      type: expanded.type,
      properties: expanded.properties,
    };
  }
  return undefined;
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
  if (!exists && Object.keys(properties).length === 0) {
    const branches = combinatorBranches(field);
    if (branches.length > 0) return expandCombinator(field, segments, index);
  }
  if (!exists && field.additionalProperties === false) return;

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
