import { beforeEach, describe, expect, it, vi } from "vitest";

const signJWTMock = vi.fn();
vi.mock("better-auth/plugins/jwt", () => ({
  signJWT: (...args: unknown[]) => signJWTMock(...args),
}));

const fakeAuthContext = { adapter: {}, secretConfig: "better-auth-secret" };
vi.mock("../auth", () => ({
  auth: { $context: Promise.resolve(fakeAuthContext) },
}));

import {
  FRONTEND_SERVICE_SUBJECT,
  mintServiceToken,
  SERVICE_TOKEN_AUDIENCE,
} from "../service-token";

interface CapturedSignArgs {
  options: { jwks: { keyPairConfig: { alg: string } } };
  payload: {
    sub: string;
    aud: string;
    scope: string;
    iat: number;
    exp: number;
  };
}

beforeEach(() => {
  signJWTMock.mockReset();
  signJWTMock.mockResolvedValue("signed-service-token");
});

describe("mintServiceToken", () => {
  it("signs with the Better Auth context and returns the token", async () => {
    const token = await mintServiceToken("auth-email:send");

    expect(token).toBe("signed-service-token");
    expect(signJWTMock).toHaveBeenCalledTimes(1);
    const [ctx] = signJWTMock.mock.calls[0] as [{ context: unknown }];
    expect(ctx.context).toBe(fakeAuthContext);
  });

  it("mints a 60-second token with the service subject, audience, and scope", async () => {
    await mintServiceToken("auth-email:send");

    const [, config] = signJWTMock.mock.calls[0] as [unknown, CapturedSignArgs];
    expect(config.payload.sub).toBe(FRONTEND_SERVICE_SUBJECT);
    expect(config.payload.aud).toBe(SERVICE_TOKEN_AUDIENCE);
    expect(config.payload.scope).toBe("auth-email:send");
    expect(config.payload.exp - config.payload.iat).toBe(60);
  });

  it("pins the key algorithm to ES256 to match the jwt plugin config", async () => {
    await mintServiceToken("auth-email:send");

    const [, config] = signJWTMock.mock.calls[0] as [unknown, CapturedSignArgs];
    expect(config.options.jwks.keyPairConfig.alg).toBe("ES256");
  });
});
