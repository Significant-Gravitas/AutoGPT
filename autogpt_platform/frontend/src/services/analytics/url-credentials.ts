import { REDACTED, redactCredentialPaths } from "@/lib/sentry-url-redaction";
import type { CaptureResult } from "posthog-js";

/**
 * PostHog `before_send` hook: masks the credentials in every URL the event
 * holds, in values and keys alike, and keeps every other parameter.
 */
export function stripURLCredentials(
  event: CaptureResult | null,
): CaptureResult | null {
  if (!event) return event;
  return {
    ...event,
    properties: stripObject(event.properties),
    ...(event.$set && { $set: stripObject(event.$set) }),
    ...(event.$set_once && { $set_once: stripObject(event.$set_once) }),
  };
}

/** Mask credential query and fragment values, and credential path segments. */
export function stripCredentialsFromURL(text: string, depth = 0): string {
  if (!text.includes("=") && !text.includes("/")) return text;
  const stripped = text.replace(
    QUERY_PARAM,
    (match, separator: string, name: string, value: string) => {
      if (isCredentialParam(name)) return `${separator}${name}=${REDACTED}`;
      const nested = stripEncodedURL(value, depth);
      return nested === value ? match : `${separator}${name}=${nested}`;
    },
  );
  return redactCredentialPaths(stripped);
}

// /reset-password's `token`, the preference links' signed `t`, the OAuth
// pages' `code` and `state`, Telegram's signed `hash` on the linking page.
// posthog-js masks these exact names in the first page it stores.
export const CREDENTIAL_PARAMS = [
  "token",
  "t",
  "code",
  "state",
  "sig",
  "hash",
  "access_token",
  "refresh_token",
  "id_token",
];
// The names a new link's secret would get.
const CREDENTIAL_PARAM_PART = /token|secret|password|signature|credential/i;
// A value runs to the next separator, so posthog-js's `<masked>` goes whole.
const QUERY_PARAM = /([?&#])([^=&#?/\s"'<>]+)=([^&#\s"']*)/g;
const MAX_NESTING = 3;
const SNAPSHOT_DATA = "$snapshot_data";

function isCredentialParam(name: string) {
  return (
    CREDENTIAL_PARAMS.includes(name.toLowerCase()) ||
    CREDENTIAL_PARAM_PART.test(name)
  );
}

// The login redirect carries the page it came from, credentials and all,
// encoded in `next`.
function stripEncodedURL(value: string, depth: number) {
  if (depth >= MAX_NESTING || !value.includes("%")) return value;
  let decoded: string;
  try {
    decoded = decodeURIComponent(value);
  } catch {
    return value;
  }
  const stripped = stripCredentialsFromURL(decoded, depth + 1);
  return stripped === decoded ? value : encodeURIComponent(stripped);
}

function stripObject(object: Record<string, unknown>) {
  const stripped: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(object)) {
    stripped[stripCredentialsFromURL(key)] =
      key === SNAPSHOT_DATA && Array.isArray(value)
        ? value.map(stripSnapshotEvent)
        : stripValue(value);
  }
  return stripped;
}

// A replay event posthog-js has gzipped carries `cv`; its strings are binary.
function stripSnapshotEvent(event: unknown) {
  return isRecord(event) && "cv" in event ? event : stripValue(event);
}

function stripValue(value: unknown): unknown {
  if (typeof value === "string") return stripCredentialsFromURL(value);
  if (Array.isArray(value)) return value.map(stripValue);
  if (isRecord(value)) return stripObject(value);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    Object.getPrototypeOf(value) === Object.prototype
  );
}
