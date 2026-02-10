import type { RJSFSchema } from "@rjsf/utils";

export function buildInputSchema(inputs: unknown): RJSFSchema | null {
  if (!inputs || typeof inputs !== "object") return null;
  const properties = inputs as RJSFSchema["properties"];
  if (!properties || Object.keys(properties).length === 0) return null;
  return inputs as RJSFSchema;
}
