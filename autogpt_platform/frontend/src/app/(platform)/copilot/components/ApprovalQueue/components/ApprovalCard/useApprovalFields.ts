import { useReviewBlockSchema } from "@/components/organisms/PendingReviewCard/components/ReviewInputFields/useReviewBlockSchema";
import type { ApprovalItem } from "../../helpers";

// A block's inputs are labelled by the block's own schema; a tool's by the server.
export function useApprovalFields(item: ApprovalItem) {
  const { schema } = useReviewBlockSchema(item.blockId);
  if (!item.blockId) return { labels: item.fields, values: item.args };
  const values = asRecord(item.args.input_data);
  const properties = asRecord(schema?.properties);
  const required = Array.isArray(schema?.required)
    ? (schema.required as string[])
    : [];
  const keys = [
    ...required.filter((key) => key in properties),
    ...Object.keys(properties).filter((key) => !required.includes(key)),
  ];
  return {
    labels: keys.map((key) => ({
      key,
      label: String(asRecord(properties[key]).title ?? key),
    })),
    values,
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}
