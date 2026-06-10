import { SignJWT } from "jose";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseSupabaseSessionCookie,
  verifyLegacyToken,
} from "../supabase-bridge";

function encodeSession(session: object): string {
  return encodeURIComponent(JSON.stringify(session));
}

describe("parseSupabaseSessionCookie", () => {
  it("extracts the access token from a plain JSON cookie", () => {
    const header = `sb-localhost-auth-token=${encodeSession({ access_token: "jwt-abc", refresh_token: "r" })}`;

    const result = parseSupabaseSessionCookie(header);

    expect(result.accessToken).toBe("jwt-abc");
    expect(result.cookieNames).toEqual(["sb-localhost-auth-token"]);
  });

  it("decodes base64-prefixed session cookies", () => {
    const payload = Buffer.from(
      JSON.stringify({ access_token: "jwt-b64" }),
    ).toString("base64");
    const header = `sb-myproject-auth-token=base64-${payload}`;

    const result = parseSupabaseSessionCookie(header);

    expect(result.accessToken).toBe("jwt-b64");
  });

  it("reassembles chunked cookies in order", () => {
    const raw = encodeSession({ access_token: "jwt-chunked" });
    const mid = Math.floor(raw.length / 2);
    const header = [
      `sb-proj-auth-token.1=${raw.slice(mid)}`,
      `sb-proj-auth-token.0=${raw.slice(0, mid)}`,
      "other=1",
    ].join("; ");

    const result = parseSupabaseSessionCookie(header);

    expect(result.accessToken).toBe("jwt-chunked");
    expect(result.cookieNames).toEqual([
      "sb-proj-auth-token.0",
      "sb-proj-auth-token.1",
    ]);
  });

  it("returns null when no supabase auth cookie is present", () => {
    const result = parseSupabaseSessionCookie(
      "better-auth.session_token=tok; theme=dark",
    );

    expect(result.accessToken).toBeNull();
    expect(result.cookieNames).toEqual([]);
  });

  it("ignores unrelated sb- cookies that are not auth tokens", () => {
    const result = parseSupabaseSessionCookie("sb-proj-refresh-meta=zzz");

    expect(result.accessToken).toBeNull();
    expect(result.cookieNames).toEqual([]);
  });

  it("reports cookie names even when the payload is malformed", () => {
    const result = parseSupabaseSessionCookie(
      "sb-proj-auth-token=not-valid-json",
    );

    expect(result.accessToken).toBeNull();
    expect(result.cookieNames).toEqual(["sb-proj-auth-token"]);
  });

  it("returns null access token when the session has none", () => {
    const header = `sb-proj-auth-token=${encodeSession({ refresh_token: "only" })}`;

    const result = parseSupabaseSessionCookie(header);

    expect(result.accessToken).toBeNull();
  });
});

const LEGACY_SECRET = "legacy-supabase-jwt-secret-0123456789abcdef";

async function signLegacyToken(
  overrides: {
    sub?: string;
    audience?: string;
    issuedAt?: number;
    expiresAt?: number;
    secret?: string;
  } = {},
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  let builder = new SignJWT({})
    .setProtectedHeader({ alg: "HS256" })
    .setAudience(overrides.audience ?? "authenticated")
    .setIssuedAt(overrides.issuedAt ?? now)
    .setExpirationTime(overrides.expiresAt ?? now + 3600);

  if (overrides.sub !== undefined) {
    builder = builder.setSubject(overrides.sub);
  }

  return builder.sign(
    new TextEncoder().encode(overrides.secret ?? LEGACY_SECRET),
  );
}

describe("verifyLegacyToken", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns the subject for a valid token", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", LEGACY_SECRET);
    const token = await signLegacyToken({ sub: "user-123" });

    expect(await verifyLegacyToken(token)).toBe("user-123");
  });

  it("accepts a token that expired within the bridge tolerance window", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", LEGACY_SECRET);
    const now = Math.floor(Date.now() / 1000);
    const token = await signLegacyToken({
      sub: "user-123",
      issuedAt: now - 3 * 60 * 60,
      expiresAt: now - 2 * 60 * 60,
    });

    expect(await verifyLegacyToken(token)).toBe("user-123");
  });

  it("rejects an expired token when the tolerance window is zero", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", LEGACY_SECRET);
    vi.stubEnv("SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS", "0");
    const now = Math.floor(Date.now() / 1000);
    const token = await signLegacyToken({
      sub: "user-123",
      issuedAt: now - 3 * 60 * 60,
      expiresAt: now - 2 * 60 * 60,
    });

    expect(await verifyLegacyToken(token)).toBeNull();
  });

  it("rejects tokens minted for a different audience", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", LEGACY_SECRET);
    const token = await signLegacyToken({
      sub: "user-123",
      audience: "service_role",
    });

    expect(await verifyLegacyToken(token)).toBeNull();
  });

  it("rejects everything when no SUPABASE_JWT_SECRET is configured", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", "");
    const token = await signLegacyToken({ sub: "user-123" });

    expect(await verifyLegacyToken(token)).toBeNull();
  });

  it("rejects garbage tokens", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", LEGACY_SECRET);

    expect(await verifyLegacyToken("not-a-jwt")).toBeNull();
  });

  it("returns null when the token has no subject", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", LEGACY_SECRET);
    const token = await signLegacyToken();

    expect(await verifyLegacyToken(token)).toBeNull();
  });
});
