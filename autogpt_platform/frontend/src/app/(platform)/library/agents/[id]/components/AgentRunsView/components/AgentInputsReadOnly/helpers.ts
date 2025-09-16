import { CredentialsMetaResponseType } from "@/app/api/__generated__/models/credentialsMetaResponseType";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

export function getCredentialTypeDisplayName(type: string): string {
  const typeDisplayMap: Record<CredentialsMetaResponseType, string> = {
    [CredentialsMetaResponseType.api_key]: "API key",
    [CredentialsMetaResponseType.oauth2]: "OAuth2",
    [CredentialsMetaResponseType.user_password]: "Username/Password",
    [CredentialsMetaResponseType.host_scoped]: "Host-Scoped",
  };

  return typeDisplayMap[type as CredentialsMetaResponseType] || type;
}

export function getAgentInputFields(agent: LibraryAgent): Record<string, any> {
  const schema = agent.input_schema as unknown as {
    properties?: Record<string, any>;
  } | null;
  if (!schema || !schema.properties) return {};
  const properties = schema.properties as Record<string, any>;
  const visibleEntries = Object.entries(properties).filter(
    ([, sub]) => !sub?.hidden,
  );
  return Object.fromEntries(visibleEntries);
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
