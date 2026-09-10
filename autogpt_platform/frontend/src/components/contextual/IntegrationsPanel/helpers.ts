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
  ideogram: "Ideogram",
  jina: "Jina",
  linkedin: "LinkedIn",
  mcp: "MCP",
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

const MCP_PROVIDER = "mcp";

// Labels that route the request rather than name the service behind it.
const MCP_HOST_NOISE = new Set(["mcp", "api", "www", "server"]);

function toHostname(value: string): string | null {
  try {
    const url = new URL(value.includes("://") ? value : `https://${value}`);
    return url.hostname || null;
  } catch {
    return null;
  }
}

function mcpServiceName(value: string): string | null {
  const host = toHostname(value);
  if (!host) return null;
  // Drop the TLD, then the routing noise, so ``mcp.sentry.dev`` reads as the
  // service a person recognises rather than the URL we happen to call.
  const name = host
    .split(".")
    .filter(Boolean)
    .slice(0, -1)
    .find((label) => !MCP_HOST_NOISE.has(label));
  return name ? formatProviderName(name) : null;
}

// The credential's own name, said the way a person would. MCP credentials are
// titled after the server URL, which is the one case where the stored title is
// an address rather than a name.
export function formatCredentialName(title: string, provider: string): string {
  const stripped = stripProviderPrefix(title, provider);
  if (provider !== MCP_PROVIDER) return stripped;
  return mcpServiceName(stripped) ?? stripped;
}

// Where the credential comes from, for the line under its name.
export function formatCredentialSource(provider: string): string {
  return provider === MCP_PROVIDER
    ? "MCP server"
    : formatProviderName(provider);
}

function toCredentialView(cred: CredentialsMetaResponse): CredentialView {
  const rawTitle = cred.title ?? formatProviderName(cred.provider);
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
  const byProvider = new Map<string, CredentialView[]>();
  for (const cred of credentials) {
    const displayProvider =
      cred.provider === "codex" ? "openai" : cred.provider;
    const list = byProvider.get(displayProvider) ?? [];
    list.push(toCredentialView(cred));
    byProvider.set(displayProvider, list);
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
