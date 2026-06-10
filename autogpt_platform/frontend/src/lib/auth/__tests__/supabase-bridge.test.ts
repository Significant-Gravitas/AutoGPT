import { describe, expect, it } from "vitest";
import { parseSupabaseSessionCookie } from "../supabase-bridge";

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
