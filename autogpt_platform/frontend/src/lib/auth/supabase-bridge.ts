import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { canConsumeLegacyCookies } from "@/lib/auth/legacy-cookies";
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

  // Order chunks by their numeric suffix (base cookie = -1), so `.2` precedes
  // `.10` — a lexicographic sort would reassemble the chunks out of order once
  // a session spans more than ten cookies.
  const chunkIndex = (name: string): number => {
    const dot = name.lastIndexOf(".");
    const suffix = dot === -1 ? "" : name.slice(dot + 1);
    return /^\d+$/.test(suffix) ? Number(suffix) : -1;
  };
  const authCookies = cookies
    .filter(({ name }) => /^sb-.+-auth-token(\.\d+)?$/.test(name))
    .sort((a, b) => chunkIndex(a.name) - chunkIndex(b.name));

  if (authCookies.length === 0) {
    return { accessToken: null, cookieNames: [] };
  }

  const cookieNames = authCookies.map(({ name }) => name);
  const joined = authCookies.map(({ value }) => value).join("");
  let raw: string;
  try {
    raw = decodeURIComponent(joined);
  } catch {
    // Malformed percent-encoding in the cookie — treat as no bridgeable session
    // rather than throwing an uncaught URIError.
    return { accessToken: null, cookieNames };
  }

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
export async function verifyLegacyToken(token: string): Promise<string | null> {
  const secret = process.env.SUPABASE_JWT_SECRET;
  if (!secret) return null;

  // Parse the raw env alone (not `|| DEFAULT`): a non-numeric value would make
  // Number() -> NaN, and clockTolerance: NaN makes jose's expiry comparison
  // always-false — i.e. ANY expired legacy token would be accepted (fail-open).
  // Fall back to the default on NaN / non-positive so a config typo can't
  // silently disable the age limit.
  const parsedMaxAgeDays = Number(
    process.env.SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS,
  );
  // Accept 0 (an explicit "zero tolerance" that rejects any expired token);
  // only NaN / negative falls back to the default.
  const maxAgeDays =
    Number.isFinite(parsedMaxAgeDays) && parsedMaxAgeDays >= 0
      ? parsedMaxAgeDays
      : DEFAULT_MAX_TOKEN_AGE_DAYS;

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
          // Same guard as the auth pages — one implementation, so a bypass
          // can't be fixed in one place and left open in the other.
          const next = sanitizeAuthNext(ctx.query?.next) ?? "/";
          const loginUrl = `${ctx.context.options.baseURL || ""}/login?next=${encodeURIComponent(next)}`;
          const nextUrl = `${ctx.context.options.baseURL || ""}${next}`;

          const cookieHeader = ctx.headers?.get("cookie") || "";
          const { accessToken, cookieNames } =
            parseSupabaseSessionCookie(cookieHeader);

          if (!canConsumeLegacyCookies()) {
            // Leave the cookies intact: once the secret is configured, the
            // user's next request bridges normally instead of being stranded.
            throw ctx.redirect(loginUrl);
          }

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

          // Don't let a banned account with a still-valid legacy cookie mint a
          // fresh Better Auth session — that would bypass the admin plugin's
          // sign-in ban gate. The bridge window can outlast a ban, so enforce
          // it here too even though no users are banned today.
          const bannedUser = user as {
            banned?: boolean | null;
            banExpires?: Date | string | null;
          };
          const banActive =
            bannedUser.banned === true &&
            (!bannedUser.banExpires ||
              new Date(bannedUser.banExpires) > new Date());
          if (banActive) {
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
