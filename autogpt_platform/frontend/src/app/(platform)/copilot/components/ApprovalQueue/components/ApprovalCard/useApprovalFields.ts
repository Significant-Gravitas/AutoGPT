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
      label: String(
        asObject(properties[key])?.title ?? serverLabel.get(key) ?? key,
      ),
    })),
    values: item.args,
  };
}
