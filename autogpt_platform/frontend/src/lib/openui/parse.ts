import { createParser } from "@openuidev/lang-core";
import { catalog, MAX_SOURCE_LENGTH } from "./catalog";
import { validateWorkspace } from "./validate";
import { validateSource } from "./source-validation";
import { validateConnections } from "./validate-connections";

export function parseWorkspace(source: string) {
  if (source.length > MAX_SOURCE_LENGTH)
    throw new Error("This workspace is too large. Try a more focused request.");
  validateSource(source);
  const result = createParser(catalog.toJSONSchema()).parse(source);
  if (result.meta.incomplete)
    throw new Error("Incomplete source. Return a complete Workspace program.");
  if (result.meta.unresolved.length)
    throw new Error(
      `Unresolved references: ${result.meta.unresolved.slice(0, 5).join(", ")}. Define each referenced section.`,
    );
  if (result.meta.errors.length)
    throw new Error(
      result.meta.errors
        .slice(0, 3)
        .map((issue) => `${issue.component}${issue.path}: ${issue.message}`)
        .join("; "),
    );
  if (result.queryStatements.length || result.mutationStatements.length)
    throw new Error(
      "Query and Mutation are not supported. Retrieve data through Copilot tools before rendering.",
    );
  if (!result.root || result.root.typeName !== "Workspace")
    throw new Error(
      "Start with root = Workspace(title, description, sections).",
    );
  validateWorkspace(result.root);
  validateConnections(result.root);
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
