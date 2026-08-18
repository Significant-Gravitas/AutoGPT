import { RJSFSchema } from "@rjsf/utils";
import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";

/**
 * Drop discriminator enum options whose credential provider this user cannot
 * connect.
 *
 * A block like Code Generation declares `credentials.discriminator = "transport"`
 * plus a `discriminator_mapping` of enum value -> provider. Offering a value
 * whose provider is gated lets the user select a transport they can never
 * satisfy: the credential row then renders "Not available on your account" and
 * execution is rejected server-side. Removing the option says the same thing
 * up front.
 *
 * Only ever narrows options for providers absent from the user's provider map.
 * `list_providers` filters exactly one provider (codex, on non-entitled
 * accounts) and returns every other provider unconditionally, so LLM blocks —
 * which discriminate on `model` across eight always-present providers — keep
 * every option.
 */
export function gateDiscriminatorOptions(
  schema: RJSFSchema,
  providers: CredentialsProvidersContextType | null,
  currentValues: Record<string, unknown>,
): RJSFSchema {
  // null means the provider map has not loaded. Filtering against it would
  // briefly drop every option, so leave the schema alone until it arrives.
  if (!providers) return schema;

  const properties = schema.properties;
  if (!properties) return schema;

  const blockedByField = collectBlockedValues(properties, providers);
  if (blockedByField.size === 0) return schema;

  const gatedProperties: NonNullable<RJSFSchema["properties"]> = {
    ...properties,
  };
  let changed = false;

  for (const [fieldName, blocked] of blockedByField) {
    const fieldSchema = properties[fieldName];
    if (!isRecord(fieldSchema) || !Array.isArray(fieldSchema.enum)) continue;

    const saved = currentValues[fieldName];
    const fallback = fieldSchema.default;
    const kept = fieldSchema.enum.filter(
      (option) =>
        !blocked.has(String(option)) ||
        // Never drop the value a graph is already saved with; hiding it would
        // silently rewrite the node on the next change. Nor the schema's own
        // default, which RJSF falls back to when the node has no saved value —
        // filtering it out leaves a select whose selection is not in its
        // options. Compared as strings to match `blocked`, so a numeric enum
        // and a string-typed saved value still line up.
        (saved !== undefined && String(option) === String(saved)) ||
        (fallback !== undefined && String(option) === String(fallback)),
    );
    if (kept.length === fieldSchema.enum.length) continue;
    // An empty dropdown is worse than an unusable option.
    if (kept.length === 0) continue;

    gatedProperties[fieldName] = { ...fieldSchema, enum: kept };
    changed = true;
  }

  return changed ? { ...schema, properties: gatedProperties } : schema;
}

function collectBlockedValues(
  properties: Record<string, unknown>,
  providers: CredentialsProvidersContextType,
): Map<string, Set<string>> {
  const blockedByField = new Map<string, Set<string>>();

  for (const propSchema of Object.values(properties)) {
    if (!isRecord(propSchema)) continue;
    if (!("credentials_provider" in propSchema)) continue;

    const discriminator = propSchema.discriminator;
    const mapping = propSchema.discriminator_mapping;
    if (typeof discriminator !== "string" || !isRecord(mapping)) continue;

    for (const [value, provider] of Object.entries(mapping)) {
      if (typeof provider !== "string" || providers[provider]) continue;
      const blocked = blockedByField.get(discriminator) ?? new Set<string>();
      blocked.add(value);
      blockedByField.set(discriminator, blocked);
    }
  }

  return blockedByField;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
