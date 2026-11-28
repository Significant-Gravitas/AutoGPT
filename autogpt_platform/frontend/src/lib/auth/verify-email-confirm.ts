import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { isAPIError } from "better-auth/api";
import { decodeJwt } from "jose";
import { NextResponse } from "next/server";
import type { auth } from "./auth";
import {
  EMAIL_VERIFICATION_NOTICE_PARAM,
  getEmailVerificationCallbackURL,
} from "./email-verification";

/**
 * A GET of a sign-up or sign-in link only parks it in a cookie and shows the
 * confirm page; the page's button verifies the address and signs in.
 */
export const VERIFY_EMAIL_CONFIRM_PATH = "/auth/verify-email";

const VERIFY_LINK_PATH = "/api/auth/verify-email";
const LINK_COOKIE = "email_verification_link";
const LINK_MAX_AGE_SECONDS = 60 * 60;

export interface HeldLink {
  token: string;
  callbackURL: string | null;
}

interface CookieReader {
  get(name: string): { value: string } | undefined;
}

interface CookieWriter {
  delete(options: { name: string; path: string }): unknown;
}

type VerifyEmailAPI = Pick<typeof auth.api, "verifyEmail">;

// Answers the GET of a sign-up or sign-in link in place of Better Auth.
export function holdVerifyLink(request: Request): Response | null {
  if (process.env.AUTH_REQUIRE_EMAIL_VERIFICATION !== "true") return null;
  const url = new URL(request.url);
  if (url.pathname !== VERIFY_LINK_PATH) return null;
  const token = url.searchParams.get("token");
  if (!token || !signsIn(token)) return null;

  const response = new NextResponse(null, {
    status: 302,
    headers: {
      Location: VERIFY_EMAIL_CONFIRM_PATH,
      "Cache-Control": "private, no-store",
    },
  });
  const link = new URLSearchParams({ token });
  const callbackURL = url.searchParams.get("callbackURL");
  if (callbackURL) link.set("callbackURL", callbackURL);
  response.cookies.set(LINK_COOKIE, link.toString(), {
    httpOnly: true,
    sameSite: "lax",
    secure: isSecureSite(),
    path: VERIFY_EMAIL_CONFIRM_PATH,
    maxAge: LINK_MAX_AGE_SECONDS,
  });
  return response;
}

export function readHeldLink(cookies: CookieReader): HeldLink | null {
  const value = cookies.get(LINK_COOKIE)?.value;
  if (!value) return null;
  const link = new URLSearchParams(value);
  const token = link.get("token");
  return token ? { token, callbackURL: link.get("callbackURL") } : null;
}

export function forgetHeldLink(cookies: CookieWriter) {
  cookies.delete({ name: LINK_COOKIE, path: VERIFY_EMAIL_CONFIRM_PATH });
}

// The address the link verifies, to show on the page; the click checks it.
export function heldLinkAddress(link: HeldLink): string | null {
  try {
    const { email } = decodeJwt(link.token);
    return typeof email === "string" ? email : null;
  } catch {
    return null;
  }
}

/**
 * The confirm page's click: verify and sign in as Better Auth's GET would
 * have, and return where its redirect would have gone, so /auth/callback
 * shows the same notices for an expired or already-used link.
 */
export async function verifyHeldLink(
  api: VerifyEmailAPI,
  link: HeldLink,
  headers: Headers,
): Promise<string> {
  const target =
    sanitizeAuthNext(link.callbackURL) ?? getEmailVerificationCallbackURL();
  try {
    await api.verifyEmail({ query: { token: link.token }, headers });
    return target;
  } catch (error) {
    if (!isAPIError(error)) throw error;
    const code = (error.body as { code?: string } | undefined)?.code;
    return withError(target, code ?? "INVALID_TOKEN");
  }
}

export function expiredLinkLoginPath() {
  return `/login?${EMAIL_VERIFICATION_NOTICE_PARAM}=expired`;
}

// A change of address carries `updateTo` and is Better Auth's own two-step
// flow, which signs no one in; an undecodable token is refused as before.
function signsIn(token: string) {
  try {
    return !decodeJwt(token).updateTo;
  } catch {
    return false;
  }
}

// Better Auth marks its own cookies Secure by the same rule.
function isSecureSite() {
  const baseURL =
    process.env.BETTER_AUTH_URL || process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;
  return baseURL?.startsWith("https://") ?? false;
}

function withError(target: string, code: string) {
  const url = new URL(target, "http://localhost");
  url.searchParams.set("error", code);
  return `${url.pathname}${url.search}`;
}
