export type MCPAuthScheme = "bearer" | "basic";

const AUTHORIZATION_HEADER_PREFIX = /^authorization\s*:\s*/i;
const SUPPORTED_SCHEME_PREFIX = /^(bearer|basic)\s+(\S[\s\S]*)$/i;

/**
 * Mirrors `normalize_mcp_authorization` in `backend/blocks/mcp/client.py`: the
 * scheme word takes the whole remainder. `mcp_auth_cases.json` pins both sides.
 */
function matchSchemePrefix(
  candidate: string,
): { scheme: MCPAuthScheme; credential: string } | null {
  const match = SUPPORTED_SCHEME_PREFIX.exec(candidate);
  if (!match) return null;
  return {
    scheme: match[1].toLowerCase() as MCPAuthScheme,
    credential: match[2].trim(),
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
 * Reject a Basic value the server cannot accept: RFC 7617 puts Base64 of
 * `user:password` on the wire, and that alphabet has neither `:` nor spaces.
 * Returns an error message, or null when the value is sendable.
 */
export function validateMCPAuthCredential(
  value: string,
  scheme: MCPAuthScheme,
): string | null {
  if (scheme !== "basic") return null;

  const candidate = value
    .trim()
    .replace(AUTHORIZATION_HEADER_PREFIX, "")
    .trim();
  const credential = matchSchemePrefix(candidate)?.credential ?? candidate;
  if (credential.includes(":")) {
    return "This looks like an unencoded user:password. Basic authentication sends the Base64 of that pair — encode it first, or paste the complete Authorization header.";
  }
  if (/\s/.test(credential)) {
    return "A Basic credential is the Base64 of user:password and cannot contain spaces.";
  }
  return null;
}

/**
 * Bare credentials get the selected scheme as an explicit prefix; the selector
 * is authoritative over any recognized pasted scheme.
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

  // An unsupported complete header is passed through for the backend to reject.
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
