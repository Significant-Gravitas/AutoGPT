import { asItems, str } from "./resultHelpers";

export interface CredentialItem {
  id: string;
  provider: string;
  type: string;
  title: string | null;
  username: string | null;
  host: string | null;
  scopes: string[];
  isManaged: boolean;
  serviceName: string | null;
}

export interface CredentialInventory {
  credentials: CredentialItem[];
  provisioningComplete: boolean;
  ungrantedProviders: string[];
}

const TYPE_LABELS: Record<string, string> = {
  oauth2: "OAuth",
  api_key: "API key",
  user_password: "Username and password",
  host_scoped: "Host-scoped",
};

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function toCredentialItem(raw: Record<string, unknown>): CredentialItem | null {
  const id = str(raw, "id");
  const provider = str(raw, "provider");
  const type = str(raw, "type");
  if (!id || !provider || !type) return null;
  return {
    id,
    provider,
    type,
    title: str(raw, "title"),
    username: str(raw, "username"),
    host: str(raw, "host"),
    scopes: stringList(raw.scopes),
    isManaged: raw.is_managed === true,
    serviceName: str(raw, "service_name"),
  };
}

export function parseCredentialInventory(
  output: Record<string, unknown>,
): CredentialInventory | null {
  if (output.type !== "credential_list") return null;
  if (!Array.isArray(output.credentials)) return null;
  return {
    credentials: (asItems(output.credentials) ?? [])
      .map(toCredentialItem)
      .filter((item): item is CredentialItem => item !== null),
    provisioningComplete: output.provisioning_complete !== false,
    ungrantedProviders: stringList(output.ungranted_providers),
  };
}

export function formatProviderName(provider: string): string {
  return provider
    .split(/[_-]/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function credentialName(credential: CredentialItem): string {
  return credential.serviceName ?? formatProviderName(credential.provider);
}

export function credentialDetail(credential: CredentialItem): string {
  const scopes = credential.scopes.length;
  return [
    credential.username ?? credential.title,
    TYPE_LABELS[credential.type] ?? credential.type,
    credential.host,
    scopes > 0 ? `${scopes} scope${scopes === 1 ? "" : "s"}` : null,
    credential.isManaged ? "Managed by AutoGPT" : null,
  ]
    .filter(Boolean)
    .join(" · ");
}
