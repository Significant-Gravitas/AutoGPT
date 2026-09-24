import { useReviewBlockSchema } from "@/components/organisms/PendingReviewCard/components/ReviewInputFields/useReviewBlockSchema";
import { asObject } from "../../../ToolChain/resultHelpers";
import type { ApprovalItem } from "../../helpers";

// A block's inputs are labelled by the block's own schema; a tool's by the server.
export function useApprovalFields(item: ApprovalItem) {
  const { schema } = useReviewBlockSchema(item.blockId);
  if (!item.blockId) return { labels: item.fields, values: item.args };
  const values = asObject(item.args.input_data) ?? {};
  const properties = asObject(schema?.properties) ?? {};
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
      label: String(asObject(properties[key])?.title ?? key),
    })),
    values,
  };
}
