import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { CredentialsMetaResponseType } from "@/app/api/__generated__/models/credentialsMetaResponseType";

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
  api_key: "API Key",
  oauth2: "OAuth",
  user_password: "User/Password",
  host_scoped: "Host-scoped",
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
  ideogram: "Ideogram",
  jina: "Jina",
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

export function formatMaskedValue(credential: CredentialView): string {
  if (credential.username) return `Username: ${credential.username}`;
  if (credential.host) return credential.host;
  if (credential.type === "api_key") return "API key configured";
  if (credential.type === "oauth2") return "Connected via OAuth";
  if (credential.type === "user_password") return "Username/password set";
  return "Configured";
}

function toCredentialView(cred: CredentialsMetaResponse): CredentialView {
  return {
    id: cred.id,
    provider: cred.provider,
    type: cred.type,
    title: cred.title ?? formatProviderName(cred.provider),
    username: cred.username ?? null,
    host: cred.host ?? null,
    isManaged: cred.is_managed ?? false,
  };
}

export function groupCredentialsByProvider(
  credentials: CredentialsMetaResponse[],
): ProviderGroupView[] {
  const byProvider = new Map<string, CredentialView[]>();
  for (const cred of credentials) {
    const list = byProvider.get(cred.provider) ?? [];
    list.push(toCredentialView(cred));
    byProvider.set(cred.provider, list);
  }

  const groups: ProviderGroupView[] = [];
  for (const [provider, creds] of byProvider) {
    groups.push({
      id: provider,
      name: formatProviderName(provider),
      credentials: creds,
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
