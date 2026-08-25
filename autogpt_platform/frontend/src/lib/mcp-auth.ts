export type MCPAuthScheme = "bearer" | "basic";

const AUTHORIZATION_HEADER_PREFIX = /^authorization\s*:\s*/i;
const SUPPORTED_SCHEME_PREFIX = /^(bearer|basic)(?:\s+|$)/i;

/** Detect an explicit Basic/Bearer scheme in a pasted credential. */
export function detectMCPAuthScheme(value: string): MCPAuthScheme | null {
  const candidate = value
    .trim()
    .replace(AUTHORIZATION_HEADER_PREFIX, "")
    .trim();
  const match = SUPPORTED_SCHEME_PREFIX.exec(candidate);
  if (!match) {
    return null;
  }
  return match[1].toLowerCase() as MCPAuthScheme;
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
  const recognizedPrefix = SUPPORTED_SCHEME_PREFIX.exec(candidate);

  // Preserve complete unsupported Authorization headers so the backend can
  // reject them explicitly instead of disguising them as a Bearer token.
  if (hasAuthorizationHeader && !recognizedPrefix) return credential;

  const credentialValue = recognizedPrefix
    ? candidate.slice(recognizedPrefix[0].length).trim()
    : candidate;
  const selectedPrefix = scheme === "basic" ? "Basic" : "Bearer";
  const headerPrefix = hasAuthorizationHeader ? "Authorization: " : "";

  return `${headerPrefix}${selectedPrefix}${
    credentialValue ? ` ${credentialValue}` : ""
  }`;
}
