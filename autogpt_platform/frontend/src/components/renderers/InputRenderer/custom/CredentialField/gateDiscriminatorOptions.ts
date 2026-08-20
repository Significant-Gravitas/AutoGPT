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
    if (!isRecord(fieldSchema)) continue;

    const saved = currentValues[fieldName];
    const fallback = fieldSchema.default;
    const keep = (option: unknown) => {
      // Never drop the value a graph is already saved with; hiding it would
      // silently rewrite the node on the next change. Nor the schema's own
      // default, which RJSF falls back to when the node has no saved value —
      // filtering it out leaves a select whose selection is not in its
      // options. Compared as strings to match `blocked`, so a numeric enum
      // and a string-typed saved value still line up.
      return (
        !blocked.has(String(option)) ||
        (saved !== undefined && String(option) === String(saved)) ||
        (fallback !== undefined && String(option) === String(fallback))
      );
    };

    const gated = gateEnumNode(fieldSchema, keep);
    if (!gated) continue;

    gatedProperties[fieldName] = gated;
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

/**
 * Filter a field's enum, returning null when nothing changed.
 *
 * An optional enum serialises as `anyOf: [{enum: [...]}, {type: "null"}]` with
 * no top-level `enum` — making a field optional would otherwise silently switch
 * the gate off, leaving a gated option on offer.
 */
function gateEnumNode(
  fieldSchema: Record<string, unknown>,
  keep: (option: unknown) => boolean,
): Record<string, unknown> | null {
  const enumValues = fieldSchema.enum;
  if (Array.isArray(enumValues)) {
    const keptIndexes = enumValues.flatMap((option, index) =>
      keep(option) ? [index] : [],
    );
    if (keptIndexes.length === enumValues.length) return null;
    // An empty dropdown is worse than an unusable option.
    if (keptIndexes.length === 0) return null;

    const originalEnumNames = fieldSchema.enumNames;
    const enumNames = Array.isArray(originalEnumNames)
      ? keptIndexes.map((index) => originalEnumNames[index])
      : undefined;
    return {
      ...fieldSchema,
      enum: keptIndexes.map((index) => enumValues[index]),
      ...(enumNames ? { enumNames } : {}),
    };
  }

  if (Array.isArray(fieldSchema.anyOf)) {
    let branchChanged = false;
    const branches = fieldSchema.anyOf.map((branch) => {
      if (!isRecord(branch)) return branch;
      const gated = gateEnumNode(branch, keep);
      if (!gated) return branch;
      branchChanged = true;
      return gated;
    });
    return branchChanged ? { ...fieldSchema, anyOf: branches } : null;
  }

  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
