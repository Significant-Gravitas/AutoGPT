import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
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
  /** A vendor sign-in (MCP) rather than a block credential. */
  isSignIn: boolean;
  /** Set when the vendor also has blocks, which this sign-in does not cover. */
  blocksNote: string | null;
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
  aiml_api: "AI/ML API",
  anysearch: "AnySearch",
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

// An MCP catalog entry (backend/integrations/mcp_catalog.json) is a provider
// named "mcp_<service>" whose display_name is what <service> formats to,
// except for these. scripts/integration-logos.test.ts checks every entry.
const MCP_DISPLAY_NAMES = new Map([
  ["mcp_agentmail", "AgentMail"],
  ["mcp_aimlapi", "AI/ML API"],
  ["mcp_apollo_io", "Apollo.io"],
  ["mcp_arize", "Arize Documentation"],
  ["mcp_aws_databases", "AWS Knowledge"],
  ["mcp_azure_cosmos_db", "Azure Cosmos DB"],
  ["mcp_companyenrich", "CompanyEnrich"],
  ["mcp_context_dev", "Context.dev"],
  ["mcp_customer_io", "Customer.io (US)"],
  ["mcp_customer_io_eu", "Customer.io (EU)"],
  ["mcp_datarobot", "DataRobot"],
  ["mcp_gojiberry_ai", "Gojiberry AI"],
  ["mcp_intercom", "Intercom (US)"],
  ["mcp_intercom_eu", "Intercom (EU)"],
  ["mcp_monday", "monday.com"],
  ["mcp_openseo", "OpenSEO"],
  ["mcp_parallel", "Parallel Search"],
  ["mcp_paypal", "PayPal"],
  ["mcp_posthog", "PostHog"],
  ["mcp_superme", "SuperMe"],
]);

export function formatProviderName(slug: unknown): string {
  if (typeof slug !== "string" || slug.length === 0) return "";
  const mcpName = MCP_DISPLAY_NAMES.get(slug);
  if (mcpName) return mcpName;
  // A bare "mcp_" names no service, so it keeps its prefix.
  const name = slug.replace(/^mcp_(?=.)/, "");
  if (Object.hasOwn(PROVIDER_DISPLAY_NAME_OVERRIDES, name)) {
    return PROVIDER_DISPLAY_NAME_OVERRIDES[name];
  }
  return name
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const PROVIDER_LIST = new Intl.ListFormat("en", {
  style: "long",
  type: "conjunction",
});

/** The services in `providers`, as one phrase: "Amplitude and Linear". A
 * service named twice, such as `linear` and `mcp_linear`, is named once. */
export function formatProviderList(providers: string[]): string {
  const names = providers.map(formatProviderName).filter(Boolean);
  return PROVIDER_LIST.format([...new Set(names)]);
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

export function groupServiceIdentity(refs: ServiceRef[]) {
  const named = refs.find((ref) => ref.service_name) ?? refs[0];
  const withIcon = refs.find((ref) => ref.service_icon) ?? refs[0];
  return { name: serviceName(named), icon: serviceIcon(withIcon) };
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

const SIGN_IN_PROVIDER = "mcp";

function toCredentialView(
  cred: CredentialsMetaResponse,
  serviceLabel: string,
  hasBlocks: boolean,
): CredentialView {
  const rawTitle = cred.title ?? serviceName(cred);
  const isSignIn = cred.provider === SIGN_IN_PROVIDER;
  return {
    id: cred.id,
    provider: cred.provider,
    type: cred.type,
    title: stripProviderPrefix(rawTitle, cred.provider),
    username: cred.username ?? null,
    host: cred.host ?? null,
    isManaged: cred.is_managed ?? false,
    isSignIn,
    blocksNote:
      isSignIn && hasBlocks
        ? `${serviceLabel} blocks in workflows need their own connection.`
        : null,
  };
}

// The services a block provider covers, so a sign-in for one of them can say
// it does not connect those blocks.
export function blockServiceKeys(providers: ProviderMetadata[]): Set<string> {
  return new Set(
    providers
      .filter((item) => !item.mcp_server)
      .map((item) =>
        serviceKey({ provider: item.name, service: item.service }),
      ),
  );
}

export function groupCredentialsByProvider(
  credentials: CredentialsMetaResponse[],
  blockServices: ReadonlySet<string> = new Set(),
): ProviderGroupView[] {
  const byService = new Map<string, CredentialsMetaResponse[]>();
  for (const cred of credentials) {
    const key = serviceKey(cred);
    byService.set(key, [...(byService.get(key) ?? []), cred]);
  }

  const groups: ProviderGroupView[] = [];
  for (const [id, creds] of byService) {
    const identity = groupServiceIdentity(creds);
    groups.push({
      id,
      name: identity.name,
      logoUrl: integrationIconSrc(identity.icon) ?? undefined,
      credentials: creds.map((cred) =>
        toCredentialView(cred, identity.name, blockServices.has(id)),
      ),
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
