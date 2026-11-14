import { exportPKCS8, generateKeyPair, jwtVerify, type JWTPayload } from "jose";
import { expect, vi } from "vitest";
import {
  appleIdentity,
  authRequest,
  origin,
  prepareAppleAuth,
  type AuthHandler,
} from "./apple-auth-fixture";

export const serviceID = "com.autogpt.test.web";
export const appleOrigin = "https://appleid.apple.com";
export const appleTokenExchange = vi.fn();
let signingKey: CryptoKey;

export async function prepareAppleWebAuth() {
  await prepareAppleAuth();
  const keys = await generateKeyPair("ES256", { extractable: true });
  signingKey = keys.publicKey;
  vi.stubEnv("AUTH_APPLE_CLIENT_ID", serviceID);
  vi.stubEnv("AUTH_APPLE_TEAM_ID", "APPLETEAM1");
  vi.stubEnv("AUTH_APPLE_KEY_ID", "APPLEKEY01");
  vi.stubEnv("AUTH_APPLE_PRIVATE_KEY", await exportPKCS8(keys.privateKey));
  appleTokenExchange.mockReset();
  answerAppleCode();
}

export function answerAppleCode(
  claims: JWTPayload = {},
  options: { omitIDToken?: boolean; rejectCode?: boolean } = {},
) {
  const fetchAppleKeys = globalThis.fetch;
  vi.stubGlobal(
    "fetch",
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url !== `${appleOrigin}/auth/token`)
        return fetchAppleKeys(input, init);
      const body = new URLSearchParams(String(init?.body));
      expect(body.get("client_id")).toBe(serviceID);
      expect(body.get("grant_type")).toBe("authorization_code");
      expect(body.get("redirect_uri")).toBe(
        `${origin}/api/auth/callback/apple`,
      );
      expect(body.get("code_verifier")).toEqual(expect.any(String));
      const { payload, protectedHeader } = await jwtVerify(
        body.get("client_secret")!,
        signingKey,
        {
          issuer: "APPLETEAM1",
          subject: serviceID,
          audience: appleOrigin,
          algorithms: ["ES256"],
        },
      );
      expect(protectedHeader.kid).toBe("APPLEKEY01");
      expect(payload.exp! - payload.iat!).toBe(300);
      appleTokenExchange({ code: body.get("code"), issuedAt: payload.iat });
      if (options.rejectCode) {
        return Response.json({ error: "invalid_grant" }, { status: 400 });
      }
      return Response.json({
        id_token: options.omitIDToken
          ? undefined
          : await appleIdentity({ aud: serviceID, ...claims }),
        access_token: "apple-test-access",
        refresh_token: "apple-test-refresh",
        token_type: "Bearer",
        expires_in: 3600,
      });
    },
  );
}

export function cookiesOf(response: Response) {
  return response.headers
    .getSetCookie()
    .map((cookie) => cookie.split(";")[0])
    .join("; ");
}

export async function beginAppleBrowserLogin(handler: AuthHandler) {
  const response = await handler(
    authRequest(
      "sign-in/social",
      {
        provider: "apple",
        callbackURL: `${origin}/auth/callback`,
      },
      { Origin: origin },
    ),
  );
  expect(response.status).toBe(200);
  const authorize = new URL((await response.json()).url);
  expect(authorize.origin).toBe(appleOrigin);
  expect(authorize.searchParams.get("client_id")).toBe(serviceID);
  expect(authorize.searchParams.get("redirect_uri")).toBe(
    `${origin}/api/auth/callback/apple`,
  );
  expect(authorize.searchParams.get("response_mode")).toBe("form_post");
  expect(authorize.searchParams.get("scope")?.split(" ")).toEqual(
    expect.arrayContaining(["email", "name"]),
  );
  return {
    state: authorize.searchParams.get("state")!,
    cookie: cookiesOf(response),
  };
}

export async function finishAppleBrowserLogin(
  handler: AuthHandler,
  login: { state: string; cookie: string },
  fields: Record<string, string> = {},
) {
  const posted = await handler(
    new Request(`${origin}/api/auth/callback/apple`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Origin: appleOrigin,
      },
      body: new URLSearchParams({
        code: "apple-test-code",
        state: login.state,
        user: JSON.stringify({
          name: { firstName: "Apple", lastName: "Browser" },
        }),
        ...fields,
      }),
    }),
  );
  expect(posted.status).toBe(302);
  const callback = posted.headers.get("location")!;
  return handler(new Request(callback, { headers: { Cookie: login.cookie } }));
}
