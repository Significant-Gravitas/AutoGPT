import type { RJSFSchema } from "@rjsf/utils";
import { customValidator } from "@/components/renderers/InputRenderer/utils/custom-validator";

export function buildInputSchema(inputs: unknown): RJSFSchema | null {
  if (!inputs || typeof inputs !== "object") return null;
  const properties = inputs as RJSFSchema["properties"];
  if (!properties || Object.keys(properties).length === 0) return null;
  return inputs as RJSFSchema;
}

export function extractDefaults(schema: RJSFSchema): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  const props = schema.properties;
  if (!props || typeof props !== "object") return defaults;

  for (const [key, prop] of Object.entries(props)) {
    if (typeof prop !== "object" || prop === null) continue;
    if ("default" in prop && prop.default !== undefined) {
      defaults[key] = prop.default;
    } else if (
      "examples" in prop &&
      Array.isArray(prop.examples) &&
      prop.examples.length > 0
    ) {
      defaults[key] = prop.examples[0];
    }
  }
  return defaults;
}

export function isFormValid(
  schema: RJSFSchema,
  formData: Record<string, unknown>,
): boolean {
  const { errors } = customValidator.validateFormData(formData, schema);
  return errors.length === 0;
}
