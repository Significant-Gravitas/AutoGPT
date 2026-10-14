import { sectionText } from "./summary-views";
import type { ElementNode } from "@openuidev/lang-core";

export function workspaceSummary(root: ElementNode) {
  const summary = collectText(root).filter(Boolean).join("\n\n");
  return summary.length <= 10_000
    ? summary
    : `${summary.slice(0, 9900)}\n\n[Text summary shortened. The interactive view contains the remaining details.]`;
}

function collectText(value: unknown): string[] {
  if (typeof value === "number" || typeof value === "string")
    return [String(value)];
  if (Array.isArray(value)) return value.flatMap(collectText);
  if (!isRecord(value)) return [];
  if ("props" in value && "typeName" in value && isRecord(value.props)) {
    const props = value.props;
    const type = String(value.typeName);
    if (type === "Form")
      return [String(props.title), ...collectText(props.fields)];
    const section = sectionText(type, props);
    if (section) return section;
    if (type === "Metric")
      return [`${props.label}: ${props.value} (${props.detail})`];
    if (
      [
        "Field",
        "SelectField",
        "DateField",
        "NumberField",
        "TextAreaField",
        "ToggleField",
        "MultiSelectField",
      ].includes(type)
    ) {
      const options = Array.isArray(props.options)
        ? props.options.filter(isRecord)
        : [];
      const selected = Array.isArray(props.value) ? props.value : [props.value];
      const labels = selected
        .map(
          (item) =>
            options.find((option) => option.value === item)?.label ?? item,
        )
        .filter((item) => item != null && item !== "");
      return [
        `${props.label}: ${labels.length ? labels.join(", ") : "Unknown / not supplied"}`,
      ];
    }
    if (type === "FollowUp") return [`Next step: ${props.label}`];
    if (type === "Map" && Array.isArray(props.locations))
      return [
        String(props.title),
        String(props.description ?? ""),
        ...props.locations
          .filter(isRecord)
          .map(
            (place) =>
              `${place.name} (${place.latitude}, ${place.longitude}): ${place.detail}`,
          ),
      ];
    if (
      type === "DataTable" &&
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
    if (type === "Checklist" && Array.isArray(props.items))
      return [
        String(props.title),
        props.items
          .filter(isRecord)
          .map(
            (item) =>
              `[${item.done ? "x" : " "}] ${item.title}\n    ${item.detail}`,
          )
          .join("\n"),
      ];
    if (type === "Timeline" && Array.isArray(props.items))
      return [
        String(props.title),
        ...props.items
          .filter(isRecord)
          .map(
            (item) =>
              `${item.time}: ${item.title} (${item.status}) — ${item.detail}`,
          ),
      ];
    return collectText(props);
  }
  return Object.entries(value)
    .filter(
      ([key]) =>
        ![
          "tone",
          "name",
          "viewId",
          "message",
          "placeholder",
          "submitLabel",
        ].includes(key),
    )
    .flatMap(([, item]) => collectText(item));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
