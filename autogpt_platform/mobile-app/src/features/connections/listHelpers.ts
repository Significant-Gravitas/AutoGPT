import type {
  CredentialMeta,
  ExpertCredential,
  ExpertSetupItem,
  MCPServerInfo,
  ProviderInfo,
} from "@/lib/api/integrations";

import { connectMethod, logoID, MCP_PROVIDER } from "./helpers";

/** What connecting a service from a list does. */
export type ServiceAction =
  | { kind: "oauth"; provider: string }
  | { kind: "sheet"; sheet: "api_key" | "user_password" | "host_scoped"; provider: string }
  /** An MCP server's own sign-in; one with none falls back to a token. */
  | { kind: "mcp"; serverURL: string }
  | { kind: "mcp_token"; serverURL: string }
  | { kind: "web" };

/** One service in the "New" list: a block provider, an MCP server, or both. */
export interface ConnectableService {
  key: string;
  name: string;
  description: string | null;
  /** Every provider name filed under this service. */
  providers: string[];
  action: ServiceAction;
}

/** An account in a list, ready to show. */
export interface AccountRow {
  id: string;
  name: string;
  detail: string | null;
  logoID: string | null;
}

/** Something an expert's workflows still need, one row per service or account. */
export interface ExpertNeed {
  key: string;
  resolution: "connect" | "allow";
  providers: string[];
  credentialID: string | null;
  workflowNames: string[];
}

// The web's PROVIDER_DISPLAY_NAME_OVERRIDES.
const DISPLAY_NAMES: Record<string, string> = {
  aiml_api: "AI/ML API",
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
  mcp: "MCP",
};

export function providerDisplayName(slug: string) {
  if (DISPLAY_NAMES[slug]) return DISPLAY_NAMES[slug];
  return slug
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

/** The account's title without a leading "<Service>: ", which its row already names. */
export function stripServicePrefix(title: string, provider: string) {
  const prefix = `${providerDisplayName(provider)}: `.toLowerCase();
  return title.toLowerCase().startsWith(prefix) ? title.slice(prefix.length) : title;
}

/** Platform keys (`is_system`, or titled as one), as the web leaves them out. */
export function isSystemCredential(credential: { title: string | null; isSystem: boolean }) {
  if (credential.isSystem) return true;
  const title = credential.title?.toLowerCase() ?? "";
  return title.includes("system") || title.includes("use credits");
}

function byName(a: { name: string }, b: { name: string }) {
  return a.name.localeCompare(b.name);
}

function accountDetail(credential: CredentialMeta) {
  const title = credential.title ? stripServicePrefix(credential.title, credential.provider) : null;
  return title || credential.username || credential.host;
}

/** Your own accounts, without platform keys, sorted by service. */
export function accountRows(credentials: CredentialMeta[]): AccountRow[] {
  return credentials
    .filter((credential) => !isSystemCredential(credential))
    .map((credential) => ({
      id: credential.id,
      name: credential.serviceName ?? providerDisplayName(credential.provider),
      detail: accountDetail(credential),
      logoID: credential.serviceIcon ?? logoID(credential.provider, credential.host),
    }))
    .sort(byName);
}

export function grantedRows(granted: ExpertCredential[]): AccountRow[] {
  return granted
    .map((item) => ({
      id: item.credentialID,
      name: item.serviceName ?? providerDisplayName(item.provider),
      detail: item.title ? stripServicePrefix(item.title, item.provider) : null,
      logoID: item.serviceIcon ?? logoID(item.provider),
    }))
    .sort(byName);
}

/** Your accounts this expert may not use yet. */
export function grantableRows(credentials: CredentialMeta[], grantedIDs: string[]) {
  return accountRows(credentials).filter((row) => !grantedIDs.includes(row.id));
}

/**
 * The expert's "connect" and "allow" items, one row per service or account
 * however many workflows ask for it. Inputs and schedules are web work.
 */
export function expertNeeds(items: ExpertSetupItem[], expertID: string): ExpertNeed[] {
  const needs = new Map<string, ExpertNeed>();
  for (const item of items) {
    if (item.expertID !== expertID) continue;
    if (item.resolution !== "connect" && item.resolution !== "allow") continue;
    if (item.resolution === "allow" && !item.credentialID) continue;
    if (item.resolution === "connect" && item.providers.length === 0) continue;
    const key =
      item.resolution === "allow"
        ? `allow:${item.credentialID}`
        : `connect:${[...item.providers].sort().join(",")}`;
    const need = needs.get(key) ?? {
      key,
      resolution: item.resolution,
      providers: item.providers,
      credentialID: item.credentialID,
      workflowNames: [],
    };
    if (item.workflowName && !need.workflowNames.includes(item.workflowName)) {
      need.workflowNames.push(item.workflowName);
    }
    needs.set(key, need);
  }
  return [...needs.values()];
}

function nativeAction(provider: ProviderInfo): ServiceAction | null {
  if (provider.authTypes.length === 0) return null;
  const method = connectMethod({ provider: provider.name, types: provider.authTypes });
  switch (method) {
    case "oauth":
      return { kind: "oauth", provider: provider.name };
    case "api_key":
    case "user_password":
    case "host_scoped":
      return { kind: "sheet", sheet: method, provider: provider.name };
    case "web":
      return { kind: "web" };
    case "mcp":
      // The generic MCP server has its own row, which asks for the address.
      return null;
  }
}

function mcpAction(server: MCPServerInfo): ServiceAction {
  if (server.connectionMode !== "hosted" || !server.serverURL) return { kind: "web" };
  if (server.authMethods.includes("oauth")) return { kind: "mcp", serverURL: server.serverURL };
  if (server.authMethods.includes("bearer")) {
    return { kind: "mcp_token", serverURL: server.serverURL };
  }
  return { kind: "web" };
}

interface ServiceGroup {
  natives: ProviderInfo[];
  server: ProviderInfo | null;
  names: string[];
}

function groupAction(key: string, group: ServiceGroup) {
  // The service's own provider leads, as on the web.
  const native =
    group.natives.find((provider) => provider.name === key) ?? group.natives[0] ?? null;
  const fromNative = native ? nativeAction(native) : null;
  if (fromNative && fromNative.kind !== "web") return { lead: native, action: fromNative };
  const fromServer = group.server?.mcpServer ? mcpAction(group.server.mcpServer) : null;
  if (fromServer && fromServer.kind !== "web") return { lead: group.server, action: fromServer };
  const action = fromNative ?? fromServer;
  return action ? { lead: native ?? group.server, action } : null;
}

/**
 * The services `GET /integrations/providers` lists, one per service: a block
 * provider and its MCP server share a row, the block provider's way in first.
 * Services with no way in at all are left out.
 */
export function toServices(providers: ProviderInfo[]): ConnectableService[] {
  const groups = new Map<string, ServiceGroup>();
  for (const provider of providers) {
    if (provider.name === MCP_PROVIDER) continue;
    const key = provider.service || provider.name;
    const group = groups.get(key) ?? { natives: [], server: null, names: [] };
    if (group.names.includes(provider.name)) continue;
    group.names.push(provider.name);
    if (provider.mcpServer) group.server ??= provider;
    else group.natives.push(provider);
    groups.set(key, group);
  }
  return [...groups.entries()]
    .flatMap(([key, group]) => {
      const chosen = groupAction(key, group);
      if (!chosen) return [];
      const named = [...group.natives, group.server].find((item) => item?.serviceName);
      return [
        {
          key,
          name: named?.serviceName ?? group.server?.displayName ?? providerDisplayName(key),
          description: chosen.lead?.description ?? null,
          providers: group.names,
          action: chosen.action,
        },
      ];
    })
    .sort(byName);
}

/** The service a setup item's providers belong to, if the app knows it. */
export function serviceFor(services: ConnectableService[], providers: string[]) {
  return (
    services.find((service) => providers.some((name) => service.providers.includes(name))) ??
    null
  );
}

/** The MCP server a row's sign-in goes to, if it signs in to one. */
export function signInServerURL(action: ServiceAction | null) {
  return action?.kind === "mcp" ? action.serverURL : null;
}

/** The button a service's row shows. */
export function actionLabel(action: ServiceAction) {
  switch (action.kind) {
    case "oauth":
    case "mcp":
      return "Sign in";
    case "sheet":
      return action.sheet === "user_password" ? "Add login" : "Add key";
    case "mcp_token":
      return "Add token";
    case "web":
      return "Open";
  }
}

function normalize(text: string) {
  return text.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/** Whether any of the texts holds the query, ignoring case and accents. */
export function matchesQuery(texts: (string | null | undefined)[], query: string) {
  const wanted = normalize(query.trim());
  if (!wanted) return true;
  return texts.some((text) => !!text && normalize(text).includes(wanted));
}
