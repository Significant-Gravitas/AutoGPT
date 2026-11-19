import { assertSameSession, sessionMark } from "@/lib/auth/token";

import { ApiError, apiRequest } from "./client";

export const CREDENTIAL_TYPES = [
  "oauth2",
  "api_key",
  "user_password",
  "host_scoped",
  "device_code",
] as const;

export type CredentialType = (typeof CREDENTIAL_TYPES)[number];

/** A saved account, as `CredentialsMetaResponse` sends it. No secrets. */
export interface CredentialMeta {
  id: string;
  provider: string;
  type: CredentialType;
  title: string | null;
  username: string | null;
  scopes: string[] | null;
  /** Host pattern for host-scoped keys, server URL for MCP. */
  host: string | null;
  isSystem: boolean;
  serviceName: string | null;
  serviceIcon: string | null;
}

/** One account an expert may use (`ExpertCredentialRef`). */
export interface ExpertCredential {
  credentialID: string;
  provider: string;
  title: string;
  type: string;
  serviceName: string | null;
  /** The logo's name in the web's `public/integrations/`. */
  serviceIcon: string | null;
}

/** An official MCP server from the catalog (`MCPServerMetadata`). */
export interface MCPServerInfo {
  serverURL: string | null;
  connectionMode: "hosted" | "custom";
  authMethods: string[];
}

/** A service you can connect (`ProviderMetadata`). */
export interface ProviderInfo {
  name: string;
  displayName: string | null;
  description: string | null;
  authTypes: CredentialType[];
  /** The key a block provider and its MCP server share. */
  service: string;
  serviceName: string | null;
  mcpServer: MCPServerInfo | null;
}

/** One thing an expert's scheduled workflow still needs (`ExpertSetupItem`). */
export interface ExpertSetupItem {
  expertID: string;
  workflowName: string | null;
  /** Any one of these would do. */
  providers: string[];
  resolution: "connect" | "allow" | "inputs" | "workflow";
  /** For "allow": the account the expert may not use yet. */
  credentialID: string | null;
}

export const integrationKeys = {
  credentials: ["integrations", "credentials"] as const,
  providers: ["integrations", "providers"] as const,
  expertCredentials: (expertID: string) =>
    ["experts", expertID, "credentials"] as const,
  expertSetup: ["experts", "setup"] as const,
};

/** Where every sign-in started from the app comes back to. */
export const OAUTH_RETURN_URL = "autogpt-app://integrations/oauth";

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown) {
  return typeof value === "string" && value ? value : null;
}

export function isCredentialType(value: unknown): value is CredentialType {
  return (
    typeof value === "string" &&
    (CREDENTIAL_TYPES as readonly string[]).includes(value)
  );
}

export function parseCredentialMeta(value: unknown): CredentialMeta | null {
  const record = toRecord(value);
  const id = text(record?.id);
  const provider = text(record?.provider);
  if (!record || !id || !provider || !isCredentialType(record.type)) return null;
  return {
    id,
    provider,
    type: record.type,
    title: text(record.title),
    username: text(record.username),
    scopes: Array.isArray(record.scopes)
      ? record.scopes.filter((scope): scope is string => typeof scope === "string")
      : null,
    host: text(record.host),
    isSystem: record.is_system === true,
    serviceName: text(record.service_name),
    serviceIcon: text(record.service_icon),
  };
}

function requireCredential(value: unknown) {
  const credential = parseCredentialMeta(value);
  if (!credential) throw new ApiError(500, "The server sent an unreadable account.");
  return credential;
}

export async function listCredentials() {
  const data = await apiRequest<unknown>("/api/integrations/credentials");
  return Array.isArray(data)
    ? data.flatMap((item) => parseCredentialMeta(item) ?? [])
    : [];
}

export interface LoginStart {
  loginURL: string;
  state: string;
}

function parseLogin(value: unknown): LoginStart {
  const record = toRecord(value);
  const loginURL = text(record?.login_url);
  const state = text(record?.state_token);
  if (!loginURL || !state) {
    throw new ApiError(500, "The server sent no sign-in page.");
  }
  return { loginURL, state };
}

interface ProviderLoginOptions {
  scopes: string[];
  /** An account to widen in place instead of adding a second one. */
  credentialID?: string;
}

export async function startProviderLogin(
  provider: string,
  { scopes, credentialID }: ProviderLoginOptions,
) {
  const data = await apiRequest<unknown>(
    `/api/integrations/${encodeURIComponent(provider)}/login`,
    {
      query: {
        scopes: scopes.length ? scopes.join(",") : undefined,
        credential_id: credentialID,
        client: "native",
      },
    },
  );
  return parseLogin(data);
}

export async function finishProviderLogin(
  provider: string,
  code: string,
  state: string,
) {
  const data = await apiRequest<unknown>(
    `/api/integrations/${encodeURIComponent(provider)}/callback`,
    { method: "POST", body: { code, state_token: state } },
  );
  return requireCredential(data);
}

/** Mirrors `NO_OAUTH_CODE` in `backend/api/features/mcp/routes.py`. */
export const MCP_NO_OAUTH = "no_oauth";

export async function startMCPLogin(serverURL: string) {
  const data = await apiRequest<unknown>("/api/mcp/oauth/login", {
    method: "POST",
    body: { server_url: serverURL, client: "native" },
  });
  return parseLogin(data);
}

export async function finishMCPLogin(code: string, state: string, iss: string | null) {
  const data = await apiRequest<unknown>("/api/mcp/oauth/callback", {
    method: "POST",
    body: { code, state_token: state, ...(iss ? { iss } : {}) },
  });
  return requireCredential(data);
}

/** Tries the token on the server first, so a refused one is never saved. */
export async function storeMCPToken(serverURL: string, token: string) {
  const mark = sessionMark();
  await apiRequest<unknown>("/api/mcp/discover-tools", {
    method: "POST",
    body: { server_url: serverURL, auth_token: token },
  });
  // Never save it to an account that signed in while it was being tried.
  assertSameSession(mark);
  const data = await apiRequest<unknown>("/api/mcp/token", {
    method: "POST",
    body: { server_url: serverURL, token },
  });
  return requireCredential(data);
}

/** The bodies the web's connect forms send to `POST /{provider}/credentials`. */
export type NewCredential =
  | { type: "api_key"; title: string; api_key: string }
  | {
      type: "user_password";
      title: string;
      username: string;
      password: string;
      scope?: string;
    }
  | {
      type: "host_scoped";
      title: string;
      host: string;
      headers: Record<string, string>;
    };

export async function createCredential(provider: string, body: NewCredential) {
  const data = await apiRequest<unknown>(
    `/api/integrations/${encodeURIComponent(provider)}/credentials`,
    { method: "POST", body: { ...body, provider } },
  );
  return requireCredential(data);
}

/** Which account the chat's tools use, per provider (or `provider:host`). */
export function recordCredentialPicks(
  sessionID: string,
  selections: Record<string, string>,
) {
  return apiRequest<unknown>(
    `/api/chat/sessions/${encodeURIComponent(sessionID)}/credential-selection`,
    { method: "PUT", body: { selections } },
  );
}

function parseExpertCredentials(value: unknown): ExpertCredential[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = toRecord(item);
    const credentialID = text(record?.credential_id);
    if (!record || !credentialID) return [];
    return [
      {
        credentialID,
        provider: text(record.provider) ?? "",
        title: text(record.title) ?? "",
        type: text(record.type) ?? "",
        serviceName: text(record.service_name),
        serviceIcon: text(record.service_icon),
      },
    ];
  });
}

export async function listExpertCredentials(expertID: string) {
  const data = await apiRequest<unknown>(
    `/api/experts/${encodeURIComponent(expertID)}/credentials`,
  );
  return parseExpertCredentials(data);
}

export async function grantExpertCredentials(
  expertID: string,
  credentialIDs: string[],
) {
  const data = await apiRequest<unknown>(
    `/api/experts/${encodeURIComponent(expertID)}/credentials`,
    { method: "POST", body: { credential_ids: credentialIDs } },
  );
  return parseExpertCredentials(data);
}

/** Takes the account away from the expert; the account itself stays. */
export async function revokeExpertCredential(expertID: string, credentialID: string) {
  const data = await apiRequest<unknown>(
    `/api/experts/${encodeURIComponent(expertID)}/credentials/${encodeURIComponent(credentialID)}`,
    { method: "DELETE" },
  );
  return parseExpertCredentials(data);
}

function strings(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && !!item)
    : [];
}

function parseMCPServer(value: unknown): MCPServerInfo | null {
  const record = toRecord(value);
  const mode = record?.connection_mode;
  if (!record || (mode !== "hosted" && mode !== "custom")) return null;
  return {
    serverURL: text(record.server_url),
    connectionMode: mode,
    authMethods: strings(record.auth_methods),
  };
}

export function parseProviders(value: unknown): ProviderInfo[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = toRecord(item);
    const name = text(record?.name);
    if (!record || !name) return [];
    return [
      {
        name,
        displayName: text(record.display_name),
        description: text(record.description),
        authTypes: strings(record.supported_auth_types).filter(isCredentialType),
        service: text(record.service) ?? name,
        serviceName: text(record.service_name),
        mcpServer: parseMCPServer(record.mcp_server),
      },
    ];
  });
}

export async function listProviders() {
  return parseProviders(await apiRequest<unknown>("/api/integrations/providers"));
}

const SETUP_RESOLUTIONS = ["connect", "allow", "inputs", "workflow"] as const;

function isResolution(value: unknown): value is ExpertSetupItem["resolution"] {
  return (
    typeof value === "string" &&
    (SETUP_RESOLUTIONS as readonly string[]).includes(value)
  );
}

export function parseExpertSetupItems(value: unknown): ExpertSetupItem[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = toRecord(item);
    const expertID = text(record?.expert_id);
    if (!record || !expertID || !isResolution(record.resolution)) return [];
    return [
      {
        expertID,
        workflowName: text(record.workflow_name),
        providers: strings(record.providers),
        resolution: record.resolution,
        credentialID: text(record.credential_id),
      },
    ];
  });
}

/** What still stands between each expert's scheduled workflows and a schedule. */
export async function listExpertSetupItems() {
  return parseExpertSetupItems(await apiRequest<unknown>("/api/experts/setup"));
}
