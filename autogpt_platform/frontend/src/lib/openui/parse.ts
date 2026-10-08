import { createParser } from "@openuidev/lang-core";
import { catalog, MAX_SOURCE_LENGTH } from "./catalog";
import { validateWorkspace } from "./validate";

export function parseWorkspace(source: string) {
  if (source.length > MAX_SOURCE_LENGTH)
    throw new Error("This workspace is too large. Try a more focused request.");
  const result = createParser(catalog.toJSONSchema()).parse(source);
  if (
    !result.root ||
    result.meta.incomplete ||
    result.meta.unresolved.length ||
    result.meta.errors.length ||
    result.queryStatements.length ||
    result.mutationStatements.length ||
    result.root.typeName !== "Workspace"
  ) {
    throw new Error(
      "The response could not be rendered as a workspace. Try again with a more specific request.",
    );
  }
  validateWorkspace(result.root);
  return result.root;
}

export function workspaceText(source: string): string {
  try {
    const root = parseWorkspace(source);
    return collectText(root).join("\n\n");
  } catch {
    return "A text summary will be available when this workspace finishes rendering.";
  }
}

function collectText(value: unknown): string[] {
  if (typeof value === "number") return [String(value)];
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(collectText);
  if (typeof value !== "object" || !value) return [];
  if ("props" in value && "typeName" in value && isRecord(value.props)) {
    const props = value.props;
    if (value.typeName === "Metric")
      return [`${props.label}: ${props.value} (${props.detail})`];
    if (
      ["Field", "SelectField", "DateField", "NumberField"].includes(
        String(value.typeName),
      )
    )
      return [`${props.label}: ${props.value}`];
    if (value.typeName === "FollowUp") return [`Next step: ${props.label}`];
    if (
      ["Chart", "TrendChart", "DonutChart"].includes(String(value.typeName)) &&
      Array.isArray(props.points)
    )
      return [
        String(props.title),
        props.points
          .filter(isRecord)
          .map((point) => `${point.label}: ${point.value} ${props.unit}`)
          .join("\n"),
      ];
    if (value.typeName === "Map" && Array.isArray(props.locations))
      return [
        String(props.title),
        ...props.locations
          .filter(isRecord)
          .map(
            (place) =>
              `${place.name} (${place.latitude}, ${place.longitude}): ${place.detail}`,
          ),
      ];
    if (
      value.typeName === "DataTable" &&
      Array.isArray(props.columns) &&
      Array.isArray(props.rows)
    )
      return [
        String(props.title),
        [props.columns, ...props.rows]
          .filter(Array.isArray)
          .map((row) => row.join(" | "))
          .join("\n"),
      ];
    if (value.typeName === "Checklist" && Array.isArray(props.items))
      return [
        String(props.title),
        props.items
          .filter(isRecord)
          .map((item) => `[ ] ${item.title}\n    ${item.detail}`)
          .join("\n"),
      ];
    return collectText(props);
  }
  return Object.entries(value)
    .filter(
      ([key]) => !["tone", "name", "message", "placeholder"].includes(key),
    )
    .flatMap(([, item]) => collectText(item));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
