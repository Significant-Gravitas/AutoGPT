import { afterEach, describe, expect, it, vi } from "vitest";
import BackendAPI, { buildOAuthLoginQuery } from "../client";

describe("BackendAPI.oAuthLogin", () => {
  it("passes credentialID through to buildOAuthLoginQuery", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    const spy = vi.spyOn(api as any, "_get").mockResolvedValue({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-abc",
    });

    const result = await api.oAuthLogin("google", ["drive.file"], "cred-1");

    expect(spy).toHaveBeenCalledWith("/integrations/google/login", {
      scopes: "drive.file",
      credential_id: "cred-1",
    });
    expect(result).toEqual({
      login_url: "https://accounts.google.com/o/oauth2/auth",
      state_token: "state-abc",
    });
  });

  it("omits query when no scopes or credentialID", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    const spy = vi
      .spyOn(api as any, "_get")
      .mockResolvedValue({ login_url: "url", state_token: "tok" });

    await api.oAuthLogin("github");

    expect(spy).toHaveBeenCalledWith("/integrations/github/login", undefined);
  });
});

describe("buildOAuthLoginQuery", () => {
  it("returns undefined when called with no args", () => {
    expect(buildOAuthLoginQuery()).toBeUndefined();
  });

  it("returns undefined when scopes is empty and credentialID is absent", () => {
    // Old behavior sent `{scopes: ""}` for an empty array, which the
    // backend rejects. Pin the tighter contract.
    expect(buildOAuthLoginQuery([])).toBeUndefined();
  });

  it("joins scopes with a comma", () => {
    expect(buildOAuthLoginQuery(["drive.file", "drive.metadata"])).toEqual({
      scopes: "drive.file,drive.metadata",
    });
  });

  it("includes credential_id when provided", () => {
    expect(buildOAuthLoginQuery(undefined, "cred-1")).toEqual({
      credential_id: "cred-1",
    });
  });

  it("includes both scopes and credential_id when both are provided", () => {
    // The incremental-OAuth flow sends both: the scopes the block needs,
    // plus the credential to merge them into.
    expect(buildOAuthLoginQuery(["drive.file"], "cred-1")).toEqual({
      scopes: "drive.file",
      credential_id: "cred-1",
    });
  });

  it("ignores an empty credentialID", () => {
    expect(buildOAuthLoginQuery(["drive.file"], "")).toEqual({
      scopes: "drive.file",
    });
  });
});

describe("BackendAPI._makeClientRequest 204 handling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mockFetch(response: Response) {
    return vi.spyOn(globalThis, "fetch").mockResolvedValue(response);
  }

  it("returns null for 204 No Content without parsing body", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    mockFetch(new Response(null, { status: 204 }));

    const result = await (api as any)._makeClientRequest(
      "DELETE",
      "/library/agents/abc",
    );

    expect(result).toBeNull();
  });

  it("returns null when Content-Length is 0", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    mockFetch(
      new Response("", { status: 200, headers: { "Content-Length": "0" } }),
    );

    const result = await (api as any)._makeClientRequest("DELETE", "/x");

    expect(result).toBeNull();
  });

  it("parses JSON body for non-empty 200 responses", async () => {
    const api = new BackendAPI("http://test", "ws://test");
    mockFetch(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const result = await (api as any)._makeClientRequest("GET", "/x");

    expect(result).toEqual({ ok: true });
  });
});
