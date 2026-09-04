export type MCPAuthScheme = "bearer" | "basic";

const AUTHORIZATION_HEADER_PREFIX = /^authorization\s*:\s*/i;
const SUPPORTED_SCHEME_PREFIX = /^(bearer|basic)\s+(\S[\s\S]*)$/i;

/**
 * Match an explicit `Basic`/`Bearer` prefix, mirroring
 * `normalize_mcp_authorization` in `backend/blocks/mcp/client.py`.
 *
 * The remainder is everything after the first run of whitespace — the same
 * split the backend does with `value.split(None, 1)`. This previously required
 * a single `\S+` run on the theory that RFC 7235 credentials are one token68,
 * and the two ends disagreed as a result: `Bearer orgid api-key` was a
 * scheme-prefixed value to the backend and a bare credential here, so this
 * function rewrote it to `Bearer Bearer orgid api-key`. A multi-word remainder
 * is not a valid Basic credential, but `validateMCPAuthCredential` is what
 * rejects that — the split itself has to agree across the boundary.
 *
 * `mcp_auth_cases.json` is the shared table both test suites assert against.
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
 * Reject a Basic value the server could not possibly accept.
 *
 * RFC 7617 puts Base64 of `user:password` on the wire, not the pair itself, and
 * the Base64 alphabet contains neither `:` nor whitespace. Both mistakes are
 * worth catching here rather than at the server:
 *
 * - a raw `pk-lf-abc:sk-lf-xyz` (what a provider's docs show) would be stored
 *   and sent verbatim, and every call would 401 with nothing pointing at the
 *   missing encoding step;
 * - a value with a space would be re-read by the backend as a bare Bearer
 *   credential, so `Basic a b` goes on the wire as `Bearer Basic a b`.
 *
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
