import { createHash, randomBytes } from "node:crypto";
import type { BetterAuthPlugin } from "better-auth";
import {
  APIError,
  createAuthEndpoint,
  getAuthoritativeSessionFromCtx,
} from "better-auth/api";
import { setSessionCookie } from "better-auth/cookies";
import {
  isMobileAuthSessionBlocked,
  MOBILE_AUTH_CALLBACK,
  MOBILE_AUTH_TTL_SECONDS,
  mobileAuthConsentPath,
  mobileAuthAuthorizeSchema,
  mobileAuthExchangeSchema,
  mobileAuthRequestSchema,
} from "./mobile-auth-helpers";

function sha256(value: string) {
  return createHash("sha256").update(value).digest("base64url");
}

function ticketIdentifier(code: string, challenge: string) {
  return `mobile-auth:${sha256(code)}:${challenge}`;
}

function invalidSession() {
  return new APIError("UNAUTHORIZED", {
    code: "MOBILE_AUTH_SESSION_REQUIRED",
    message: "Sign in again in the browser to connect AutoGPT.",
  });
}

function blockedSession() {
  return new APIError("FORBIDDEN", {
    code: "MOBILE_AUTH_NOT_ALLOWED",
    message: "This account cannot connect a mobile app session.",
  });
}

function requireMobileOrigin(
  origin: string | null | undefined,
  baseURL: string,
) {
  if (origin !== new URL(baseURL).origin) {
    throw new APIError("FORBIDDEN", {
      code: "MOBILE_AUTH_INVALID_ORIGIN",
      message: "Start sign-in from the AutoGPT app.",
    });
  }
}

export function mobileAuth() {
  return {
    id: "mobile-auth",
    rateLimit: [
      {
        pathMatcher: (path) => path.startsWith("/mobile/"),
        window: 60,
        max: 20,
      },
    ],
    endpoints: {
      startMobileAuth: createAuthEndpoint(
        "/mobile/start",
        { method: "GET", query: mobileAuthRequestSchema },
        async (ctx) => {
          ctx.setHeader("Cache-Control", "no-store");
          ctx.setHeader("Referrer-Policy", "no-referrer");
          const consentPath = mobileAuthConsentPath(ctx.query);
          const session = await getAuthoritativeSessionFromCtx(ctx);
          const path = session
            ? consentPath
            : `/login?next=${encodeURIComponent(consentPath)}`;
          const origin = new URL(ctx.context.baseURL).origin;
          throw ctx.redirect(`${origin}${path}`);
        },
      ),
      authorizeMobileAuth: createAuthEndpoint(
        "/mobile/authorize",
        { method: "POST", body: mobileAuthAuthorizeSchema },
        async (ctx) => {
          requireMobileOrigin(ctx.headers?.get("origin"), ctx.context.baseURL);
          ctx.setHeader("Cache-Control", "no-store");
          ctx.setHeader("Referrer-Policy", "no-referrer");
          const session = await getAuthoritativeSessionFromCtx(ctx);
          if (!session) throw invalidSession();
          if (
            isMobileAuthSessionBlocked(
              session,
              ctx.context.options.emailAndPassword?.requireEmailVerification,
            )
          ) {
            throw blockedSession();
          }
          if (session.user.id !== ctx.body.expected_user_id) {
            throw new APIError("FORBIDDEN", {
              code: "MOBILE_AUTH_ACCOUNT_CHANGED",
              message:
                "Your browser account changed. Restart sign-in in the app.",
            });
          }
          const code = randomBytes(32).toString("base64url");
          await ctx.context.internalAdapter.createVerificationValue({
            identifier: ticketIdentifier(code, ctx.body.code_challenge),
            value: session.session.token,
            expiresAt: new Date(Date.now() + MOBILE_AUTH_TTL_SECONDS * 1000),
          });
          const callback = new URL(MOBILE_AUTH_CALLBACK);
          callback.searchParams.set("code", code);
          callback.searchParams.set("state", ctx.body.state);
          return ctx.json({ url: callback.toString() });
        },
      ),
      exchangeMobileAuth: createAuthEndpoint(
        "/mobile/exchange",
        {
          method: "POST",
          body: mobileAuthExchangeSchema,
          metadata: {
            allowedMediaTypes: [
              "application/json",
              "application/x-www-form-urlencoded",
            ],
          },
        },
        async (ctx) => {
          requireMobileOrigin(ctx.headers?.get("origin"), ctx.context.baseURL);
          ctx.setHeader("Cache-Control", "no-store");
          ctx.setHeader("Referrer-Policy", "no-referrer");
          const identifier = ticketIdentifier(
            ctx.body.code,
            sha256(ctx.body.code_verifier),
          );
          const ticket =
            await ctx.context.internalAdapter.consumeVerificationValue(
              identifier,
            );
          if (!ticket) {
            throw new APIError("BAD_REQUEST", {
              code: "MOBILE_AUTH_INVALID_CODE",
              message: "This sign-in request expired or was already used.",
            });
          }
          const source = await ctx.context.internalAdapter.findSession(
            ticket.value,
          );
          if (!source || source.session.expiresAt.getTime() <= Date.now()) {
            throw invalidSession();
          }
          if (
            isMobileAuthSessionBlocked(
              source,
              ctx.context.options.emailAndPassword?.requireEmailVerification,
            )
          ) {
            throw blockedSession();
          }
          const session = await ctx.context.internalAdapter.createSession(
            source.user.id,
          );
          if (!session) throw invalidSession();
          try {
            const currentSource = await ctx.context.internalAdapter.findSession(
              ticket.value,
            );
            if (
              !currentSource ||
              currentSource.session.expiresAt.getTime() <= Date.now()
            ) {
              throw invalidSession();
            }
            if (
              isMobileAuthSessionBlocked(
                currentSource,
                ctx.context.options.emailAndPassword?.requireEmailVerification,
              )
            ) {
              throw blockedSession();
            }
            await setSessionCookie(ctx, { session, user: currentSource.user });
          } catch (error) {
            await ctx.context.internalAdapter.deleteSession(session.token);
            throw error;
          }
          return ctx.json({ success: true });
        },
      ),
    },
  } satisfies BetterAuthPlugin;
}
