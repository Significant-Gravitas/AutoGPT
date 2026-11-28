import { CredentialsMetaResponseType } from "@/app/api/__generated__/models/credentialsMetaResponseType";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

export function getCredentialTypeDisplayName(type: string): string {
  const typeDisplayMap: Record<CredentialsMetaResponseType, string> = {
    [CredentialsMetaResponseType.api_key]: "API key",
    [CredentialsMetaResponseType.oauth2]: "OAuth2",
    [CredentialsMetaResponseType.user_password]: "Username/Password",
    [CredentialsMetaResponseType.host_scoped]: "Host-Scoped",
    [CredentialsMetaResponseType.device_code]: "Device Auth",
  };

  return typeDisplayMap[type as CredentialsMetaResponseType] || type;
}

// A triggered agent has both: its graph inputs AND the trigger block's config.
// They are stored separately on a preset, so never fall back from one to the other.
export function getAgentInputFields(agent: LibraryAgent): Record<string, any> {
  return getVisibleFields(agent.input_schema);
}

export function getTriggerConfigFields(
  agent: LibraryAgent,
): Record<string, any> {
  return getVisibleFields(agent.trigger_setup_info?.config_schema);
}

export function getVisibleFields(schema: unknown): Record<string, any> {
  const properties = (schema as { properties?: Record<string, any> } | null)
    ?.properties;
  if (!properties) return {};
  return Object.fromEntries(
    Object.entries(properties).filter(([, sub]) => !sub?.hidden),
  );
}

export function getAgentCredentialsFields(
  agent: LibraryAgent,
): Record<string, any> {
  if (
    !agent.credentials_input_schema ||
    typeof agent.credentials_input_schema !== "object" ||
    !("properties" in agent.credentials_input_schema) ||
    !agent.credentials_input_schema.properties
  ) {
    return {};
  }
  return agent.credentials_input_schema.properties as Record<string, any>;
}

/**
 * The schema a field shows the key it held with. A field open to several
 * providers, like an LLM step's whose model comes from a link, shows the key
 * under its own provider: without the discriminator, which goes by the step's
 * other inputs, there is no other provider to show it under.
 */
export function getHeldKeySchema<
  Schema extends { credentials_provider?: string[] },
>(schema: Schema, credential: { provider: string }): Schema {
  const providers = schema.credentials_provider ?? [];
  return {
    ...schema,
    discriminator: undefined,
    credentials_provider:
      providers.length > 1 ? [credential.provider] : providers,
  };
}

export function renderValue(value: any): string {
  if (value === undefined || value === null) return "";
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  )
    return String(value);
  try {
    return JSON.stringify(value, undefined, 2);
  } catch {
    return String(value);
  }
}
