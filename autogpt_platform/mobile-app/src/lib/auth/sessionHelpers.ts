// No native imports, so tests can load this without a phone.

export interface SessionUser {
  id: string;
  email: string;
  name: string;
  image?: string | null;
}

export interface StoredSession {
  /** The signed session credential: only the auth service ever gets it. */
  token: string;
  user: SessionUser;
}

/** Why a sign-in was refused. `status` is the server's answer. */
export interface SignInFailure {
  status: number;
  code?: string;
  message?: string;
}

export const NO_APP_SESSION_CODE = "NO_APP_SESSION";

const SESSION_COOKIE = "better-auth.session_token";

/** The server's origin in characters a SecureStore key takes. */
export function serverSlug(serverURL: string) {
  // SecureStore keys take letters, digits, ".", "-" and "_" only. Hex keeps
  // every origin apart: replacing ":" with "_" would make http://dev:3000 and
  // http://dev_3000 share one key.
  return Array.from(new URL(serverURL).origin, (char) =>
    char.charCodeAt(0).toString(16).padStart(2, "0"),
  ).join("");
}

/** One saved session per server, so switching servers never sends one's credential to another. */
export function storageKey(serverURL: string) {
  return `autogpt.session.${serverSlug(serverURL)}`;
}

// The verification link opens the web, which creates the platform user and
// records consent, then lands on the page that sends the person back here.
const VERIFIED_PATH = "/app-verified";

/** Where the verification link leads, as the web builds it (lib/auth/email-verification.ts). */
export function verificationCallbackURL(marketingOptOut: boolean) {
  const params = new URLSearchParams({ method: "email", next: VERIFIED_PATH });
  if (marketingOptOut) params.set("marketing_opt_out", "1");
  return `/auth/callback?${params.toString()}`;
}

/** The name a new account starts with, as the web names it. */
export function nameFromEmail(email: string) {
  return email.split("@")[0] ?? email;
}

export type SignUpOutcome = "signedIn" | "verify" | "noAppSession";

/**
 * What a 2xx sign-up answer means. `signedToken` is the `set-auth-token`
 * header: the credential a server with the installed-app contract hands over.
 */
export function readSignUpOutcome(
  body: unknown,
  signedToken: string | null,
): SignUpOutcome {
  if (signedToken && readUser(body)) return "signedIn";
  // With verification on there is no session until the link is opened.
  const token = (body as { token?: unknown } | null)?.token;
  if (token === null && !signedToken) return "verify";
  return "noAppSession";
}

export function parseStoredSession(raw: string | null): StoredSession | null {
  if (!raw) return null;
  try {
    const saved = JSON.parse(raw) as { token?: unknown; user?: unknown };
    const user = readUser(saved);
    if (typeof saved.token !== "string" || !saved.token || !user) return null;
    return { token: saved.token, user };
  } catch {
    return null;
  }
}

/** The user in a sign-in or session answer, or null when there is none. */
export function readUser(body: unknown): SessionUser | null {
  const user = (body as { user?: Record<string, unknown> } | null)?.user;
  if (!user || typeof user.id !== "string" || typeof user.email !== "string") {
    return null;
  }
  return {
    id: user.id,
    email: user.email,
    name: typeof user.name === "string" ? user.name : "",
    image: typeof user.image === "string" ? user.image : null,
  };
}

export async function readSignInFailure(
  response: Response,
): Promise<SignInFailure> {
  const body = (await response.json().catch(() => null)) as {
    code?: unknown;
    message?: unknown;
  } | null;
  return {
    status: response.status,
    code: typeof body?.code === "string" ? body.code : undefined,
    message: typeof body?.message === "string" ? body.message : undefined,
  };
}

/**
 * The session as the cookie the web app's own pages and routes read. The
 * signed credential is that cookie's value; an https site names it with the
 * __Secure- prefix.
 */
export function sessionCookieHeader(token: string, serverURL: string) {
  const isSecure = new URL(serverURL).protocol === "https:";
  const name = isSecure ? `__Secure-${SESSION_COOKIE}` : SESSION_COOKIE;
  return `${name}=${encodeURIComponent(decode(token))}`;
}

// The server may hand the credential over already percent-encoded.
function decode(token: string) {
  try {
    return decodeURIComponent(token);
  } catch {
    return token;
  }
}
