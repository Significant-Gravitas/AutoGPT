import type { BetterAuthPlugin } from "better-auth";
import { createAuthEndpoint } from "better-auth/api";
import { setSessionCookie } from "better-auth/cookies";
import { jwtVerify } from "jose";

const DEFAULT_MAX_TOKEN_AGE_DAYS = 30;

/**
 * Picks the Supabase auth-token cookies (including chunked `.0`/`.1` parts)
 * out of a Cookie header and reassembles the stored session JSON.
 */
export function parseSupabaseSessionCookie(cookieHeader: string): {
  accessToken: string | null;
  cookieNames: string[];
} {
  const cookies = cookieHeader
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const eq = part.indexOf("=");
      return { name: part.slice(0, eq), value: part.slice(eq + 1) };
    });

  const authCookies = cookies
    .filter(({ name }) => /^sb-.+-auth-token(\.\d+)?$/.test(name))
    .sort((a, b) => a.name.localeCompare(b.name));

  if (authCookies.length === 0) {
    return { accessToken: null, cookieNames: [] };
  }

  const cookieNames = authCookies.map(({ name }) => name);
  let raw = decodeURIComponent(authCookies.map(({ value }) => value).join(""));

  if (raw.startsWith("base64-")) {
    try {
      raw = Buffer.from(raw.slice("base64-".length), "base64").toString(
        "utf-8",
      );
    } catch {
      return { accessToken: null, cookieNames };
    }
  }

  try {
    const session = JSON.parse(raw);
    const accessToken = session?.access_token;
    return {
      accessToken: typeof accessToken === "string" ? accessToken : null,
      cookieNames,
    };
  } catch {
    return { accessToken: null, cookieNames };
  }
}

/**
 * Verifies a legacy Supabase (GoTrue) HS256 access token. Expired tokens are
 * accepted within the configured window: the signature still proves the
 * cookie came from a real GoTrue login, and GoTrue sessions outlived their
 * 1-hour access tokens via refresh tokens that no longer have an issuer to
 * talk to. Without this tolerance only users active in the final hour before
 * cutover would keep their sessions.
 */
async function verifyLegacyToken(token: string): Promise<string | null> {
  const secret = process.env.SUPABASE_JWT_SECRET;
  if (!secret) return null;

  const maxAgeDays = Number(
    process.env.SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS ||
      DEFAULT_MAX_TOKEN_AGE_DAYS,
  );

  try {
    const { payload } = await jwtVerify(
      token,
      new TextEncoder().encode(secret),
      {
        algorithms: ["HS256"],
        audience: "authenticated",
        clockTolerance: maxAgeDays * 24 * 60 * 60,
      },
    );
    return typeof payload.sub === "string" ? payload.sub : null;
  } catch {
    return null;
  }
}

/**
 * Better Auth plugin that silently upgrades a legacy Supabase session into a
 * Better Auth session, so existing logged-in users don't get kicked to the
 * login page by the auth migration.
 *
 * GET /api/auth/supabase-bridge?next=<path> — called by the middleware when a
 * request carries Supabase auth cookies but no Better Auth session cookie.
 * Either way the legacy cookies are cleared so the bridge runs at most once.
 */
export function supabaseBridge() {
  return {
    id: "supabase-bridge",
    endpoints: {
      bridgeSupabaseSession: createAuthEndpoint(
        "/supabase-bridge",
        { method: "GET" },
        async (ctx) => {
          const rawNext = ctx.query?.next;
          // Only same-origin relative paths to prevent open redirects.
          const next =
            typeof rawNext === "string" &&
            rawNext.startsWith("/") &&
            !rawNext.startsWith("//")
              ? rawNext
              : "/";
          const loginUrl = `${ctx.context.options.baseURL || ""}/login?next=${encodeURIComponent(next)}`;
          const nextUrl = `${ctx.context.options.baseURL || ""}${next}`;

          const cookieHeader = ctx.headers?.get("cookie") || "";
          const { accessToken, cookieNames } =
            parseSupabaseSessionCookie(cookieHeader);

          for (const name of cookieNames) {
            ctx.setCookie(name, "", { path: "/", maxAge: 0 });
          }

          if (!accessToken) {
            throw ctx.redirect(loginUrl);
          }

          const userId = await verifyLegacyToken(accessToken);
          if (!userId) {
            throw ctx.redirect(loginUrl);
          }

          const user = await ctx.context.internalAdapter.findUserById(userId);
          if (!user) {
            throw ctx.redirect(loginUrl);
          }

          const session = await ctx.context.internalAdapter.createSession(
            user.id,
          );
          await setSessionCookie(ctx, { session, user });

          throw ctx.redirect(nextUrl);
        },
      ),
    },
  } satisfies BetterAuthPlugin;
}
