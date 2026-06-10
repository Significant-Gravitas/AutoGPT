import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const signInSocialMock = vi.fn();
const isWaitlistErrorMock = vi.fn();
const logWaitlistErrorMock = vi.fn();

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      signInSocial: (...args: unknown[]) => signInSocialMock(...args),
    },
  },
}));

vi.mock("@/app/api/auth/utils", () => ({
  isWaitlistError: (...args: unknown[]) => isWaitlistErrorMock(...args),
  logWaitlistError: (...args: unknown[]) => logWaitlistErrorMock(...args),
}));

import { POST } from "../route";

function makeProviderRequest(body: unknown): Request {
  return new Request("http://localhost:3000/api/auth/provider", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

beforeEach(() => {
  signInSocialMock.mockReset();
  isWaitlistErrorMock.mockReset().mockReturnValue(false);
  logWaitlistErrorMock.mockReset();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("POST /api/auth/provider", () => {
  it("returns the social sign-in URL and forwards redirectTo as the callback URL", async () => {
    signInSocialMock.mockResolvedValue({
      url: "https://accounts.google.com/o/oauth2/auth?state=xyz",
    });

    const request = makeProviderRequest({
      provider: "google",
      redirectTo: "/auth/callback?next=/marketplace",
    });
    const response = await POST(request);

    expect(signInSocialMock).toHaveBeenCalledWith({
      body: {
        provider: "google",
        callbackURL: "/auth/callback?next=/marketplace",
      },
      headers: request.headers,
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      url: "https://accounts.google.com/o/oauth2/auth?state=xyz",
    });
  });

  it("falls back to the default callback URL when redirectTo is omitted", async () => {
    vi.stubEnv("AUTH_CALLBACK_URL", "");
    signInSocialMock.mockResolvedValue({ url: "https://github.com/oauth" });

    await POST(makeProviderRequest({ provider: "github" }));

    expect(signInSocialMock).toHaveBeenCalledWith({
      body: { provider: "github", callbackURL: "/auth/callback" },
      headers: expect.any(Headers),
    });
  });

  it("returns 400 when no provider is supplied", async () => {
    const response = await POST(makeProviderRequest({ redirectTo: "/x" }));

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "Invalid provider" });
    expect(signInSocialMock).not.toHaveBeenCalled();
  });

  it("returns 403 not_allowed for waitlist rejections", async () => {
    isWaitlistErrorMock.mockReturnValue(true);
    signInSocialMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: 'The email address "[email]" is not allowed to register.',
        code: "P0001",
      }),
    );

    const response = await POST(makeProviderRequest({ provider: "google" }));

    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({ error: "not_allowed" });
    expect(logWaitlistErrorMock).toHaveBeenCalledWith(
      "OAuth Provider",
      expect.any(String),
    );
  });

  it("returns 400 with the Better Auth message for other APIErrors", async () => {
    signInSocialMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: "Provider not configured",
        code: "PROVIDER_NOT_FOUND",
      }),
    );

    const response = await POST(makeProviderRequest({ provider: "discord" }));

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "Provider not configured" });
  });

  it("returns 500 when sign-in fails with an unexpected error", async () => {
    signInSocialMock.mockRejectedValue(new Error("connection refused"));

    const response = await POST(makeProviderRequest({ provider: "google" }));

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({
      error: "Failed to initiate OAuth",
    });
  });

  it("returns 500 when the request body is not valid JSON", async () => {
    const request = new Request("http://localhost:3000/api/auth/provider", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{not json",
    });

    const response = await POST(request);

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({
      error: "Failed to initiate OAuth",
    });
  });
});
