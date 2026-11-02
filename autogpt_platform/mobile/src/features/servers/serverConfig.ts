import { stableDigest } from "../../lib/ids";

/**
 * The boundary every network call goes through. `authUrl` is the origin of
 * the Better Auth service (the web app's origin on a standard deployment),
 * `apiUrl` the origin of the REST backend. The session credential only ever
 * goes to `authUrl`; short-lived access tokens only to `apiUrl`.
 *
 * Discovery from a single entered address is planned server work; until it
 * exists an own server is described by both origins explicitly.
 */
export interface ServerConfig {
  id: string;
  name: string;
  authUrl: string;
  apiUrl: string;
  kind: "cloud" | "custom";
}

/**
 * Hosted AutoGPT. The web app (and its Better Auth routes) is served from
 * platform.agpt.co and the API from backend.agpt.co: see the repository
 * README and .oss-scanner/threat_model.md for the former, and
 * frontend/CONTRIBUTING.md (production OpenAPI) for the latter. Sign-in only
 * succeeds once that deployment issues signed native sessions; until then
 * the sign-in screen reports the server as not yet supporting the app.
 */
export const CLOUD_SERVER: ServerConfig = {
  id: "cloud",
  name: "AutoGPT Cloud",
  authUrl: "https://platform.agpt.co",
  apiUrl: "https://backend.agpt.co",
  kind: "cloud",
};

export type OriginResult =
  | { ok: true; origin: string }
  | { ok: false; reason: string };

/**
 * Accepts `host`, `https://host[:port]` or a trailing slash, and returns a
 * bare origin. Paths, queries, fragments and embedded credentials are
 * refused. Plain HTTP is only allowed for this device's loopback addresses,
 * which is what a simulator reaching a local development server uses.
 */
export function normalizeOrigin(input: string): OriginResult {
  const trimmed = input.trim();
  if (!trimmed) return { ok: false, reason: "Enter an address." };
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;
  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    return { ok: false, reason: "That isn't a valid web address." };
  }
  if (url.username || url.password) {
    return {
      ok: false,
      reason: "Remove the username or password from the address.",
    };
  }
  if ((url.pathname && url.pathname !== "/") || url.search || url.hash) {
    return {
      ok: false,
      reason: "Enter only the server address, without a path.",
    };
  }
  if (!url.hostname)
    return { ok: false, reason: "That address has no host name." };
  if (url.protocol === "http:") {
    if (!isLoopback(url.hostname)) {
      return {
        ok: false,
        reason: "Use an https:// address. Plain http only works for localhost.",
      };
    }
  } else if (url.protocol !== "https:") {
    return { ok: false, reason: "Use an https:// address." };
  }
  return { ok: true, origin: url.origin };
}

function isLoopback(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, "").toLowerCase();
  return (
    host === "localhost" || host === "::1" || /^127(\.\d{1,3}){3}$/.test(host)
  );
}

export function hostLabel(origin: string): string {
  try {
    return new URL(origin).host;
  } catch {
    return origin;
  }
}

export async function createCustomServer(args: {
  name: string;
  authUrl: string;
  apiUrl: string;
}): Promise<ServerConfig> {
  const id = `custom-${await stableDigest(`${args.authUrl}\n${args.apiUrl}`)}`;
  return {
    id,
    name: args.name.trim() || hostLabel(args.authUrl),
    authUrl: args.authUrl,
    apiUrl: args.apiUrl,
    kind: "custom",
  };
}

/** Partitions credentials and caches: the same origins always share one key. */
export async function serverPartitionKey(
  server: ServerConfig,
): Promise<string> {
  return stableDigest(`${server.authUrl}\n${server.apiUrl}`);
}

export function isServerConfig(value: unknown): value is ServerConfig {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.name === "string" &&
    typeof v.authUrl === "string" &&
    typeof v.apiUrl === "string" &&
    (v.kind === "cloud" || v.kind === "custom") &&
    normalizeOrigin(v.authUrl).ok &&
    normalizeOrigin(v.apiUrl).ok
  );
}
