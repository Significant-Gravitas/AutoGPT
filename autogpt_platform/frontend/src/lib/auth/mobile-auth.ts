import type { BetterAuthPlugin } from "better-auth";
import {
  createAuthEndpoint,
  getAuthoritativeSessionFromCtx,
} from "better-auth/api";
import { setSessionCookie } from "better-auth/cookies";
import { createHash, randomBytes } from "node:crypto";
import { z } from "zod";
import { mobileConsentPage } from "./mobile-consent-page";

const startSchema = z.object({
  challenge: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
  state: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
});

export function mobileAuth() {
  return {
    id: "mobile-auth",
    endpoints: {
      startMobileAuth: createAuthEndpoint(
        "/mobile/start",
        { method: "GET", query: startSchema },
        async (ctx) => {
          ctx.setHeader("Cache-Control", "no-store");
          const origin = new URL(ctx.context.baseURL).origin;
          const session = await getAuthoritativeSessionFromCtx(ctx);
          const next = `/api/auth/mobile/start?${new URLSearchParams(ctx.query)}`;
          if (!session) {
            throw ctx.redirect(
              `${origin}/login?${new URLSearchParams({ next })}`,
            );
          }
          return mobileConsentPage(session.user.email, ctx.query);
        },
      ),
      approveMobileAuth: createAuthEndpoint(
        "/mobile/approve",
        {
          method: "POST",
          body: startSchema,
          metadata: {
            allowedMediaTypes: ["application/x-www-form-urlencoded"],
          },
        },
        async (ctx) => {
          const origin = new URL(ctx.context.baseURL).origin;
          if (ctx.headers?.get("origin") !== origin) {
            throw ctx.error("FORBIDDEN", { message: "Invalid origin" });
          }
          const session = await getAuthoritativeSessionFromCtx(ctx);
          if (!session) throw ctx.error("UNAUTHORIZED");
          const code = randomBytes(32).toString("base64url");
          await ctx.context.internalAdapter.createVerificationValue({
            identifier: grantIdentifier(code, ctx.body.challenge),
            value: session.session.token,
            expiresAt: new Date(Date.now() + 60_000),
          });
          ctx.setHeader("Cache-Control", "no-store");
          ctx.setHeader("Referrer-Policy", "no-referrer");
          throw ctx.redirect(
            `autogpt://auth/callback?${new URLSearchParams({
              code,
              state: ctx.body.state,
            })}`,
          );
        },
      ),
      exchangeMobileAuth: createAuthEndpoint(
        "/mobile/exchange",
        {
          method: "POST",
          metadata: {
            allowedMediaTypes: ["application/x-www-form-urlencoded"],
          },
          body: z.object({
            code: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
            verifier: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
          }),
        },
        async (ctx) => {
          ctx.setHeader("Cache-Control", "no-store");
          const challenge = digest(ctx.body.verifier);
          const grant =
            await ctx.context.internalAdapter.consumeVerificationValue(
              grantIdentifier(ctx.body.code, challenge),
            );
          if (!grant)
            throw ctx.error("UNAUTHORIZED", {
              message: "Sign-in expired. Please try again.",
            });
          const source = await ctx.context.internalAdapter.findSession(
            grant.value,
          );
          if (!source || source.session.expiresAt <= new Date()) {
            throw ctx.error("UNAUTHORIZED");
          }
          const user = source.user as typeof source.user & {
            banned?: boolean | null;
            banExpires?: Date | string | null;
          };
          if (
            user.banned &&
            (!user.banExpires || new Date(user.banExpires) > new Date())
          ) {
            throw ctx.error("FORBIDDEN");
          }
          const session = await ctx.context.internalAdapter.createSession(
            user.id,
          );
          await setSessionCookie(ctx, { session, user });
          throw ctx.redirect(`${new URL(ctx.context.baseURL).origin}/home`);
        },
      ),
    },
    rateLimit: [
      {
        pathMatcher: (path) => path.startsWith("/mobile/"),
        window: 60,
        max: 20,
      },
    ],
  } satisfies BetterAuthPlugin;
}

function digest(value: string) {
  return createHash("sha256").update(value).digest("base64url");
}

function grantIdentifier(code: string, challenge: string) {
  return `mobile-auth:${digest(code)}:${challenge}`;
}
