import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";

// An optional input is `anyOf: [X, {type: "null"}]`; `determineDataType`
// classifies it by X, so the options must be read from X too.
export function getSelectOptions(schema: BlockIOSubSchema): string[] {
  const branch =
    "anyOf" in schema ? schema.anyOf.find((s) => s.type === "string") : schema;
  return branch && "enum" in branch ? (branch.enum ?? []) : [];
}

export function getMultiSelectProperties(
  schema: BlockIOSubSchema,
): Record<string, BlockIOSubSchema> {
  const branch =
    "anyOf" in schema ? schema.anyOf.find((s) => s.type === "object") : schema;
  return branch && "properties" in branch ? branch.properties : {};
}
