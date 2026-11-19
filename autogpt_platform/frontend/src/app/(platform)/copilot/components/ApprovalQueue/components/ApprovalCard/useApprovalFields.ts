import { useReviewBlockSchema } from "@/components/organisms/PendingReviewCard/components/ReviewInputFields/useReviewBlockSchema";
import { asObject } from "../../../ToolChain/resultHelpers";
import type { ApprovalItem } from "../../helpers";

// A block's inputs are labelled and ordered by the block's own schema; a tool's by the server.
export function useApprovalFields(item: ApprovalItem) {
  const { schema } = useReviewBlockSchema(item.blockId);
  const properties = asObject(schema?.properties);
  if (!item.blockId || !properties) {
    return { labels: item.fields, values: item.args };
  }
  const required = Array.isArray(schema?.required)
    ? (schema.required as string[])
    : [];
  const keys = [
    ...required.filter((key) => key in properties),
    ...Object.keys(properties).filter((key) => !required.includes(key)),
    ...Object.keys(item.args).filter((key) => !(key in properties)),
  ];
  const serverLabel = new Map(item.fields.map((f) => [f.key, f.label]));
  return {
    labels: keys.map((key) => ({
      key,
      label: schemaLabel(
        key,
        asObject(properties[key])?.title,
        serverLabel.get(key) ?? key,
      ),
    })),
    values: item.args,
  };
}

// A title a person wrote wins; pydantic's own (the key in Title Case, or a type's name) does not.
function schemaLabel(key: string, title: unknown, fallback: string) {
  if (typeof title !== "string" || !title) return fallback;
  const automatic = key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
  const typeName = /^[A-Z][A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*(\[.*\])?$/;
  if (title === automatic || typeName.test(title)) return fallback;
  return title;
}
