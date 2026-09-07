import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { test } from "node:test";
import { createFixtureServer } from "./server.mjs";

const verifier =
  "native-fixture-verifier-0123456789-abcdefghijklmnopqrstuvwxyz"; // pragma: allowlist secret
const state = "fixture-state-0123456789-abcdefghijklmnopqrstuvwxyz"; // pragma: allowlist secret
const challenge = createHash("sha256").update(verifier).digest("base64url");

async function startFixture(t, options = {}) {
  const server = createFixtureServer(options);
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise((resolve) => server.close(resolve)));
  return `http://127.0.0.1:${server.address().port}`;
}

async function startAuth(origin, overrides = {}) {
  const query = new URLSearchParams({
    code_challenge: challenge,
    state,
    ...overrides,
  });
  return fetch(`${origin}/api/auth/mobile/start?${query}`, {
    redirect: "manual",
  });
}

async function exchange(origin, code, codeVerifier = verifier) {
  return fetch(`${origin}/api/auth/mobile/exchange`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: origin },
    body: JSON.stringify({ code, code_verifier: codeVerifier }),
  });
}

async function authorize(origin) {
  const start = await startAuth(origin);
  assert.equal(start.status, 302);
  const consent = await fetch(new URL(start.headers.get("location"), origin));
  assert.equal(consent.status, 200);
  assert.match(await consent.text(), /Connect AutoGPT/);
  const response = await fetch(`${origin}/api/auth/mobile/authorize`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: origin },
    body: JSON.stringify({ code_challenge: challenge, state }),
  });
  assert.equal(response.status, 200);
  return new URL((await response.json()).url);
}

test("native callback exchanges PKCE for a persistent HttpOnly fixture session", async (t) => {
  const origin = await startFixture(t);
  const callback = await authorize(origin);
  assert.equal(
    `${callback.protocol}//${callback.host}${callback.pathname}`,
    "autogpt://auth/callback",
  );
  assert.equal(callback.searchParams.get("state"), state);
  const exchanged = await exchange(origin, callback.searchParams.get("code"));
  assert.equal(exchanged.status, 200);
  assert.deepEqual(await exchanged.json(), { success: true });
  const setCookies = exchanged.headers.getSetCookie();
  assert.equal(setCookies.length, 2);
  assert.match(
    setCookies[1],
    /^better-auth\.session_data=.*Expires=[A-Z][a-z]{2}, /,
  );
  const cookieHeader = setCookies
    .map((cookie) => cookie.split(";")[0])
    .join("; ");
  const setCookie = setCookies[0];
  assert.match(setCookie, /^better-auth\.session_token=/);
  assert.match(setCookie, /HttpOnly/);
  assert.match(setCookie, /Max-Age=86400/);
  const session = await fetch(`${origin}/api/fixture/session`, {
    headers: { Cookie: cookieHeader },
  });
  const sessionResult = await session.json();
  assert.equal(sessionResult.authenticated, true);
  assert.equal(sessionResult.cacheCookieReceived, true);
  const page = await fetch(`${origin}/copilot`, {
    headers: { Cookie: cookieHeader },
  });
  assert.match(await page.text(), /Native integration fixture/);
});

test("a successful authorization code cannot be replayed", async (t) => {
  const origin = await startFixture(t);
  const callback = await authorize(origin);
  const code = callback.searchParams.get("code");
  assert.equal((await exchange(origin, code)).status, 200);
  assert.equal((await exchange(origin, code)).status, 400);
});

test("an expired authorization code cannot establish a session", async (t) => {
  let now = 1000;
  const origin = await startFixture(t, { now: () => now, codeLifetimeMs: 100 });
  const callback = await authorize(origin);
  now += 101;
  const response = await exchange(origin, callback.searchParams.get("code"));
  assert.equal(response.status, 400);
  assert.equal(response.headers.get("set-cookie"), null);
});

test("a mismatched verifier cannot establish a session", async (t) => {
  const origin = await startFixture(t);
  const callback = await authorize(origin);
  const response = await exchange(
    origin,
    callback.searchParams.get("code"),
    "different-verifier-0123456789-abcdefghijklmnopqrstuvwxyz", // pragma: allowlist secret
  );
  assert.equal(response.status, 400);
  assert.equal(response.headers.get("set-cookie"), null);
});

test("start rejects incomplete PKCE input and ignores callback override", async (t) => {
  const origin = await startFixture(t);
  assert.equal(
    (await startAuth(origin, { code_challenge: "short" })).status,
    400,
  );
  assert.equal((await startAuth(origin, { state: "" })).status, 400);
  const response = await startAuth(origin, {
    redirect_uri: "https://example.com",
  });
  assert.equal(
    new URL(response.headers.get("location"), origin).pathname,
    "/auth/mobile",
  );
  assert.equal(
    new URL(response.headers.get("location"), origin).searchParams.has(
      "redirect_uri",
    ),
    false,
  );
});

test("fixtures expose deterministic download, HTTP error, and transport failure probes", async (t) => {
  const origin = await startFixture(t);
  const download = await fetch(`${origin}/attachment.txt`);
  assert.match(download.headers.get("content-disposition"), /attachment/);
  assert.match(await download.text(), /Native integration fixture/);
  assert.equal((await fetch(`${origin}/error`)).status, 503);
  await assert.rejects(fetch(`${origin}/offline`), /fetch failed/);
});
