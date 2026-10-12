import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";

export function getSelectOptions(schema: BlockIOSubSchema) {
  if (!("enum" in schema) || !Array.isArray(schema.enum)) return [];
  return schema.enum.filter((option) => option);
}
