import type { CredentialField } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";

const VALID_CREDENTIAL_TYPES = new Set([
  "api_key",
  "oauth2",
  "user_password",
  "host_scoped",
]);

/**
 * Transforms raw missing_credentials from SetupRequirementsResponse
 * into CredentialField[] tuples compatible with CredentialsGroupedView.
 *
 * Each CredentialField is [key, schema] where schema matches
 * BlockIOCredentialsSubSchema shape.
 */
export function coerceCredentialFields(rawMissingCredentials: unknown): {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
} {
  const missing =
    rawMissingCredentials && typeof rawMissingCredentials === "object"
      ? (rawMissingCredentials as Record<string, unknown>)
      : {};

  const credentialFields: CredentialField[] = [];
  const requiredCredentials = new Set<string>();

  Object.entries(missing).forEach(([key, value]) => {
    if (!value || typeof value !== "object") return;
    const cred = value as Record<string, unknown>;

    const provider =
      typeof cred.provider === "string" ? cred.provider.trim() : "";
    if (!provider) return;

    const types =
      Array.isArray(cred.types) && cred.types.length > 0
        ? cred.types
        : typeof cred.type === "string"
          ? [cred.type]
          : [];

    const credentialTypes = types
      .map((t) => (typeof t === "string" ? t.trim() : ""))
      .filter((t) => VALID_CREDENTIAL_TYPES.has(t));

    if (credentialTypes.length === 0) return;

    const scopes = Array.isArray(cred.scopes)
      ? cred.scopes.filter((s): s is string => typeof s === "string")
      : undefined;

    const schema = {
      type: "object" as const,
      properties: {},
      credentials_provider: [provider],
      credentials_types: credentialTypes,
      credentials_scopes: scopes,
    };

    credentialFields.push([key, schema]);
    requiredCredentials.add(key);
  });

  return { credentialFields, requiredCredentials };
}

export function coerceExpectedInputs(rawInputs: unknown): Array<{
  name: string;
  title: string;
  type: string;
  description?: string;
  required: boolean;
}> {
  if (!Array.isArray(rawInputs)) return [];
  const results: Array<{
    name: string;
    title: string;
    type: string;
    description?: string;
    required: boolean;
  }> = [];

  rawInputs.forEach((value, index) => {
    if (!value || typeof value !== "object") return;
    const input = value as Record<string, unknown>;

    const name =
      typeof input.name === "string" && input.name.trim()
        ? input.name.trim()
        : `input-${index}`;
    const title =
      typeof input.title === "string" && input.title.trim()
        ? input.title.trim()
        : name;
    const type = typeof input.type === "string" ? input.type : "unknown";
    const description =
      typeof input.description === "string" && input.description.trim()
        ? input.description.trim()
        : undefined;
    const required = Boolean(input.required);

    const item: {
      name: string;
      title: string;
      type: string;
      description?: string;
      required: boolean;
    } = { name, title, type, required };
    if (description) item.description = description;
    results.push(item);
  });

  return results;
}
