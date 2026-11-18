import {
  isCredentialType,
  type CredentialMeta,
  type CredentialType,
} from "@/lib/api/integrations";
import { FRONTEND_URL } from "@/lib/env";

/** An account the user already has that an expert could be allowed to use. */
export interface Grantable {
  id: string;
  title: string;
  type: string;
}

/** Why an account the user has doesn't do (`credential_gaps.py`). */
export interface CredentialGap {
  kind: "missing_scopes" | "reconnect_required";
  credentialID: string;
  credentialTitle: string | null;
}

/** One service a tool needs, from a `missing_credentials` entry. */
export interface CredentialRequest {
  /** The tool's field name; the key the backend uses for this need. */
  key: string;
  provider: string;
  providerName: string;
  types: CredentialType[];
  scopes: string[];
  discriminatorValues: string[];
  gap: CredentialGap | null;
  expertGrant: { expertID: string; credentials: Grantable[] } | null;
}

/** How a service gets connected from the phone. */
export type ConnectMethod =
  | "oauth"
  | "mcp"
  | "api_key"
  | "user_password"
  | "host_scoped"
  | "web";

/**
 * AutoGPT holds the Ayrshare account (linking happens on Ayrshare's page) and
 * Codex signs in with a device code: both are still web-only.
 */
const WEB_ONLY_PROVIDERS = ["ayrshare", "codex"];

export const MCP_PROVIDER = "mcp";

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function strings(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && !!item)
    : [];
}

function text(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function titleCase(slug: string) {
  return slug
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

function parseGap(value: unknown): CredentialGap | null {
  const record = toRecord(value);
  const credentialID = text(record?.credential_id);
  const kind = record?.kind;
  if (!credentialID || (kind !== "missing_scopes" && kind !== "reconnect_required")) {
    return null;
  }
  return { kind, credentialID, credentialTitle: text(record?.credential_title) };
}

function parseExpertGrant(value: unknown): CredentialRequest["expertGrant"] {
  const record = toRecord(value);
  const expertID = text(record?.expert_id);
  if (!expertID) return null;
  const credentials = Array.isArray(record?.credentials)
    ? record.credentials.flatMap((entry) => {
        const credential = toRecord(entry);
        const id = text(credential?.id);
        if (!id) return [];
        return [
          {
            id,
            title: text(credential?.title) ?? id,
            type: text(credential?.type) ?? "api_key",
          },
        ];
      })
    : [];
  return { expertID, credentials };
}

function parseRequest(key: string, value: unknown): CredentialRequest | null {
  const record = toRecord(value);
  const provider = text(record?.provider);
  if (!record || !provider) return null;
  const types = strings(record.types).filter(isCredentialType);
  if (types.length === 0) return null;
  return {
    key,
    provider,
    providerName: text(record.provider_name) ?? titleCase(provider),
    types,
    scopes: strings(record.scopes),
    discriminatorValues: strings(record.discriminator_values),
    gap: parseGap(record.credential_gap),
    expertGrant: parseExpertGrant(record.expert_grant),
  };
}

/** Reads `missing_credentials`, dropping entries the app can't act on. */
export function parseCredentialRequests(raw: unknown): CredentialRequest[] {
  const record = toRecord(raw);
  if (!record) return [];
  return Object.entries(record).flatMap(([key, value]) => parseRequest(key, value) ?? []);
}

export function connectMethod(
  request: Pick<CredentialRequest, "provider" | "types">,
): ConnectMethod {
  if (request.provider === MCP_PROVIDER) return "mcp";
  if (WEB_ONLY_PROVIDERS.includes(request.provider)) return "web";
  if (request.types.includes("oauth2")) return "oauth";
  if (request.types.includes("api_key")) return "api_key";
  if (request.types.includes("user_password")) return "user_password";
  if (request.types.includes("host_scoped")) return "host_scoped";
  return "web";
}

/** Mirrors the backend's `normalize_mcp_url`. */
export function normalizeServerURL(url: string) {
  return url.trim().replace(/\/+$/, "");
}

export function hostFromURL(url: string) {
  const match = /^(?:[a-z][a-z\d+.-]*:\/\/)?([^/?#:]+)/i.exec(url.trim());
  return match ? match[1].toLowerCase() : null;
}

/** The single host a host-scoped need names, as the web keys its picks. */
export function requestHost(request: CredentialRequest) {
  if (!request.types.includes("host_scoped")) return null;
  const hosts = [
    ...new Set(request.discriminatorValues.flatMap((url) => hostFromURL(url) ?? [])),
  ];
  return hosts.length === 1 ? hosts[0] : null;
}

/** The MCP server a need is for. */
export function requestServerURL(request: CredentialRequest, fallback: string | null) {
  const url = request.discriminatorValues[0] ?? fallback;
  return url ? normalizeServerURL(url) : null;
}

function hasScopes(credential: CredentialMeta, scopes: string[]) {
  if (credential.type !== "oauth2" || scopes.length === 0) return true;
  const granted = new Set(credential.scopes ?? []);
  return scopes.every((scope) => granted.has(scope));
}

function matchesTarget(credential: CredentialMeta, request: CredentialRequest) {
  const wanted = request.discriminatorValues;
  if (credential.provider === MCP_PROVIDER) {
    return (
      credential.host !== null &&
      wanted.map(normalizeServerURL).includes(normalizeServerURL(credential.host))
    );
  }
  if (credential.type === "host_scoped" && credential.host) {
    return (
      wanted.length === 0 ||
      wanted.some((url) => hostFromURL(url) === credential.host?.toLowerCase())
    );
  }
  return true;
}

/** Whether this account does what the need asks, as the web's matcher decides. */
export function credentialFits(credential: CredentialMeta, request: CredentialRequest) {
  return (
    !credential.isSystem &&
    credential.provider === request.provider &&
    request.types.includes(credential.type) &&
    hasScopes(credential, request.scopes) &&
    matchesTarget(credential, request)
  );
}

export function fittingCredentials(
  credentials: CredentialMeta[],
  request: CredentialRequest,
  excludedIDs: string[] = [],
) {
  return credentials.filter(
    (credential) =>
      !excludedIDs.includes(credential.id) && credentialFits(credential, request),
  );
}

/** What to show under an account's name. */
export function credentialLabel(credential: { title: string | null; username?: string | null }) {
  return credential.title ?? credential.username ?? null;
}

export function webChatURL(sessionID: string | null) {
  return sessionID
    ? `${FRONTEND_URL}/copilot?sessionId=${encodeURIComponent(sessionID)}`
    : `${FRONTEND_URL}/copilot`;
}

/** One or two letters for a service's tile: "GitHub" → "GH", "Google Calendar" → "GC". */
export function tileLetters(name: string) {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length > 1) return (words[0][0] + words[1][0]).toUpperCase();
  const capitals = words[0].replace(/[^A-Z]/g, "");
  return capitals.length >= 2 ? capitals.slice(0, 2) : words[0][0].toUpperCase();
}

/** A stable pick from `choices` for a name, so a service keeps its colour. */
export function pickByName<T>(name: string, choices: readonly T[]) {
  let hash = 0;
  for (const char of name.toLowerCase()) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return choices[hash % choices.length];
}
