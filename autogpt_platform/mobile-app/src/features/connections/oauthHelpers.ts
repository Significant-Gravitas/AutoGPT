import { OAUTH_RETURN_URL } from "@/lib/api/integrations";

/** What a sign-in started from the app signs in to. */
export type OAuthTarget =
  | {
      kind: "provider";
      provider: string;
      providerName: string;
      scopes: string[];
      /** An account to widen in place; the backend keeps its scopes. */
      credentialID?: string;
    }
  | { kind: "mcp"; serverURL: string; providerName: string };

/** Where a sign-in was started, so the result can find its way back. */
export interface FlowOrigin {
  sessionID: string | null;
  /**
   * The setup card (its tool call id) and the row on it. Null for a sign-in
   * from the chat panel, which in an expert's chat allows the new account.
   */
  cardKey: string | null;
  rowKey: string | null;
  expertID: string | null;
}

/** A sign-in in the system browser; saved so a cold start can finish it. */
export interface PendingFlow extends FlowOrigin {
  state: string;
  target: OAuthTarget;
  savedAt: number;
}

/** What the web's callback page sends back to `autogpt-app://integrations/oauth`. */
export interface OAuthReturn {
  state: string;
  code: string | null;
  iss: string | null;
  error: string | null;
  errorDescription: string | null;
}

/** A pending flow older than this is dropped; the provider's code has expired anyway. */
export const PENDING_FLOW_MAX_AGE_MS = 15 * 60_000;

function param(value: unknown) {
  const first = Array.isArray(value) ? value[0] : value;
  return typeof first === "string" && first ? first : null;
}

/** Reads the return from route params (or any key/value record). */
export function oauthReturnFromParams(
  params: Record<string, unknown>,
): OAuthReturn | null {
  const state = param(params.state);
  if (!state) return null;
  return {
    state,
    code: param(params.code),
    iss: param(params.iss),
    error: param(params.error),
    errorDescription: param(params.error_description),
  };
}

/** Reads the URL the auth session ended on; null for any other URL. */
export function parseOAuthReturn(url: string): OAuthReturn | null {
  if (!url.startsWith(OAUTH_RETURN_URL)) return null;
  const rest = url.slice(OAUTH_RETURN_URL.length);
  if (rest && !rest.startsWith("?") && !rest.startsWith("#")) return null;
  const query = rest.replace(/^[?#]/, "").split("#")[0];
  const params: Record<string, string> = {};
  for (const pair of query.split("&")) {
    if (!pair) continue;
    const [rawKey, ...rawValue] = pair.split("=");
    try {
      const key = decodeURIComponent(rawKey.replace(/\+/g, " "));
      params[key] = decodeURIComponent(rawValue.join("=").replace(/\+/g, " "));
    } catch {
      return null;
    }
  }
  return oauthReturnFromParams(params);
}

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nullableText(value: unknown) {
  return typeof value === "string" && value ? value : null;
}

function parseTarget(value: unknown): OAuthTarget | null {
  const record = toRecord(value);
  const providerName = nullableText(record?.providerName);
  if (!record || !providerName) return null;
  if (record.kind === "mcp") {
    const serverURL = nullableText(record.serverURL);
    return serverURL ? { kind: "mcp", serverURL, providerName } : null;
  }
  const provider = nullableText(record.provider);
  if (record.kind !== "provider" || !provider) return null;
  return {
    kind: "provider",
    provider,
    providerName,
    scopes: Array.isArray(record.scopes)
      ? record.scopes.filter((scope): scope is string => typeof scope === "string")
      : [],
    credentialID: nullableText(record.credentialID) ?? undefined,
  };
}

/** The saved flow, or null when it is missing, unreadable or too old. */
export function parsePendingFlow(raw: string | null, now: number): PendingFlow | null {
  if (!raw) return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  const record = toRecord(value);
  const state = nullableText(record?.state);
  const target = parseTarget(record?.target);
  const savedAt = record?.savedAt;
  if (!record || !state || !target || typeof savedAt !== "number") return null;
  if (now - savedAt > PENDING_FLOW_MAX_AGE_MS || savedAt > now) return null;
  return {
    state,
    target,
    savedAt,
    sessionID: nullableText(record.sessionID),
    cardKey: nullableText(record.cardKey),
    rowKey: nullableText(record.rowKey),
    expertID: nullableText(record.expertID),
  };
}

/** The line a provider's refusal shows; null when the user simply said no. */
export function providerErrorMessage(error: string, providerName: string) {
  if (error === "access_denied") return null;
  return `${providerName} didn't finish the sign-in. Try again.`;
}
