export type MCPAuthScheme = "bearer" | "basic";

const AUTHORIZATION_HEADER_PREFIX = /^authorization\s*:\s*/i;
const SUPPORTED_SCHEME_PREFIX = /^(bearer|basic)\s+(\S+)$/i;

/**
 * Match an explicit `Basic`/`Bearer` prefix, mirroring
 * `normalize_mcp_authorization` in `backend/blocks/mcp/client.py`.
 *
 * RFC 7235 credentials are a single token68/base64 run with no internal
 * whitespace, so a value whose remainder still contains a space is a bare
 * multi-word credential that merely starts with "basic"/"bearer" — not a
 * scheme-prefixed one. Both ends have to agree on that, or a value this
 * function rewrites gets read back with the other meaning.
 */
function matchSchemePrefix(
  candidate: string,
): { scheme: MCPAuthScheme; credential: string } | null {
  const match = SUPPORTED_SCHEME_PREFIX.exec(candidate);
  if (!match) return null;
  return {
    scheme: match[1].toLowerCase() as MCPAuthScheme,
    credential: match[2],
  };
}

/** Detect an explicit Basic/Bearer scheme in a pasted credential. */
export function detectMCPAuthScheme(value: string): MCPAuthScheme | null {
  const candidate = value
    .trim()
    .replace(AUTHORIZATION_HEADER_PREFIX, "")
    .trim();
  return matchSchemePrefix(candidate)?.scheme ?? null;
}

/**
 * Reject a credential the backend would read back as a different scheme.
 *
 * A Basic credential is Base64 of `user:password`, so whitespace inside it is
 * always a paste mistake — and one we cannot send safely: the backend treats a
 * multi-word value after a scheme word as a bare Bearer credential, so
 * `Basic a b` would be stored and sent as `Bearer Basic a b`. Fail loudly here
 * instead. Returns an error message, or null when the value is sendable.
 */
export function validateMCPAuthCredential(
  value: string,
  scheme: MCPAuthScheme,
): string | null {
  const candidate = value
    .trim()
    .replace(AUTHORIZATION_HEADER_PREFIX, "")
    .trim();
  const credential = matchSchemePrefix(candidate)?.credential ?? candidate;
  if (scheme === "basic" && /\s/.test(credential)) {
    return "A Basic credential is the Base64 of user:password and cannot contain spaces.";
  }
  return null;
}

/**
 * Prepare the existing token API payload without changing Bearer compatibility.
 * Bare credentials get the selected scheme as an explicit prefix.
 * The selector is authoritative over any recognized pasted scheme.
 */
export function prepareMCPAuthCredential(
  value: string,
  scheme: MCPAuthScheme,
): string {
  const credential = value.trim();
  if (!credential) return credential;

  const hasAuthorizationHeader = AUTHORIZATION_HEADER_PREFIX.test(credential);
  const candidate = credential.replace(AUTHORIZATION_HEADER_PREFIX, "").trim();
  const recognizedPrefix = matchSchemePrefix(candidate);

  // Preserve complete unsupported Authorization headers so the backend can
  // reject them explicitly instead of disguising them as a Bearer token.
  if (hasAuthorizationHeader && !recognizedPrefix) return credential;

  const credentialValue = recognizedPrefix
    ? recognizedPrefix.credential
    : candidate;
  const selectedPrefix = scheme === "basic" ? "Basic" : "Bearer";
  const headerPrefix = hasAuthorizationHeader ? "Authorization: " : "";

  return `${headerPrefix}${selectedPrefix}${
    credentialValue ? ` ${credentialValue}` : ""
  }`;
}
