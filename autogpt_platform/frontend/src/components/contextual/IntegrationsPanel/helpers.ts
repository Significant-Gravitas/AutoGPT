import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { CredentialsMetaResponseType } from "@/app/api/__generated__/models/credentialsMetaResponseType";
import { integrationIconSrc } from "@/components/molecules/IntegrationLogo/helpers";

export type CredentialType = CredentialsMetaResponseType;

export interface CredentialView {
  id: string;
  provider: string;
  type: CredentialType;
  title: string;
  username: string | null;
  host: string | null;
  isManaged: boolean;
}

export interface ProviderGroupView {
  id: string;
  name: string;
  logoUrl?: string;
  credentials: CredentialView[];
}

const TYPE_LABELS: Record<CredentialType, string> = {
  api_key: "API Key", // pragma: allowlist secret
  oauth2: "OAuth",
  user_password: "User/Password", // pragma: allowlist secret
  host_scoped: "Host-scoped",
  device_code: "Device auth",
};

export function typeBadgeLabel(type: CredentialType): string {
  return TYPE_LABELS[type] ?? type;
}

const PROVIDER_DISPLAY_NAME_OVERRIDES: Record<string, string> = {
  github: "GitHub",
  google: "Google",
  google_maps: "Google Maps",
  hubspot: "HubSpot",
  openai: "OpenAI",
  anthropic: "Anthropic",
  openweathermap: "OpenWeatherMap",
  e2b: "E2B",
  d_id: "D-ID",
  dataforseo: "DataForSEO",
  ideogram: "Ideogram",
  jina: "Jina",
  linkedin: "LinkedIn",
  twitter: "X",
  zerobounce: "ZeroBounce",
};

export function formatProviderName(slug: unknown): string {
  if (typeof slug !== "string" || slug.length === 0) return "";
  if (PROVIDER_DISPLAY_NAME_OVERRIDES[slug]) {
    return PROVIDER_DISPLAY_NAME_OVERRIDES[slug];
  }
  return slug
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export interface ServiceRef {
  provider: string;
  service?: string;
  service_name?: string | null;
  service_icon?: string | null;
}

// The backend names the service behind every credential and provider; these
// read it with the one display alias the backend does not know about.
export function serviceKey(ref: ServiceRef): string {
  const key = ref.service || ref.provider;
  return key === "codex" ? "openai" : key;
}

export function serviceName(ref: ServiceRef): string {
  return ref.service_name || formatProviderName(serviceKey(ref));
}

export function serviceIcon(ref: ServiceRef): string {
  return ref.service_icon || serviceKey(ref);
}

export function serviceLabelFromIcon(id: string): string {
  return id.startsWith("mcp:")
    ? id.slice("mcp:".length)
    : formatProviderName(id);
}

export function formatMaskedValue(credential: CredentialView): string {
  if (credential.username) return `Username: ${credential.username}`;
  if (credential.host) return credential.host;
  if (credential.type === "api_key") return "API key configured";
  if (credential.type === "oauth2") return "Connected via OAuth";
  if (credential.type === "user_password") return "Username/password set";
  return "Configured";
}

export function stripProviderPrefix(title: string, provider: string): string {
  // The row already lives under the provider group, so any leading
  // ``<ProviderName>: `` in the per-credential title doubles up.  Strip
  // it generically (case-insensitive) so e.g. ``"MCP: mcp.sentry.dev"``
  // collapses to ``"mcp.sentry.dev"`` without a per-provider branch.
  const displayName = formatProviderName(provider);
  if (!displayName) return title;
  const prefix = `${displayName}: `;
  return title.toLowerCase().startsWith(prefix.toLowerCase())
    ? title.slice(prefix.length)
    : title;
}

export function formatCredentialName(title: string, provider: string): string {
  return stripProviderPrefix(title, provider);
}

function toCredentialView(cred: CredentialsMetaResponse): CredentialView {
  const rawTitle = cred.title ?? serviceName(cred);
  return {
    id: cred.id,
    provider: cred.provider,
    type: cred.type,
    title: stripProviderPrefix(rawTitle, cred.provider),
    username: cred.username ?? null,
    host: cred.host ?? null,
    isManaged: cred.is_managed ?? false,
  };
}

export function groupCredentialsByProvider(
  credentials: CredentialsMetaResponse[],
): ProviderGroupView[] {
  const byService = new Map<string, CredentialsMetaResponse[]>();
  for (const cred of credentials) {
    const key = serviceKey(cred);
    byService.set(key, [...(byService.get(key) ?? []), cred]);
  }

  const groups: ProviderGroupView[] = [];
  for (const [id, creds] of byService) {
    groups.push({
      id,
      name: serviceName(creds[0]),
      logoUrl: integrationIconSrc(serviceIcon(creds[0])) ?? undefined,
      credentials: creds.map(toCredentialView),
    });
  }
  groups.sort((a, b) => a.name.localeCompare(b.name));
  return groups;
}

function normalizeSearchText(value: string): string {
  return value.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

export function filterProviders(
  providers: ProviderGroupView[],
  query: string,
): ProviderGroupView[] {
  const q = normalizeSearchText(query.trim());
  if (!q) return providers;

  const result: ProviderGroupView[] = [];
  for (const provider of providers) {
    if (normalizeSearchText(provider.name).includes(q)) {
      result.push(provider);
      continue;
    }
    const matched = provider.credentials.filter(
      (c) =>
        normalizeSearchText(c.title).includes(q) ||
        (c.username && normalizeSearchText(c.username).includes(q)),
    );
    if (matched.length > 0) {
      result.push({ ...provider, credentials: matched });
    }
  }
  return result;
}
