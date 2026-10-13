// Better Auth sign-in for the load tests.
//
// The backend only accepts the ES256 JWTs that the frontend's Better Auth
// issues at GET /api/auth/token (audience "authenticated", valid for an hour).
// setup() signs each account in once per run and gets its token; every VU
// then uses one account's token. A run therefore signs in once per account,
// however many VUs and iterations it has.
//
// Accounts come from LOAD_TEST_ACCOUNTS, a JSON array of
// {"email": "...", "password": "..."}. When that is unset and the target
// allows it (PREVIEW and LOCAL), setup() signs up LOAD_TEST_ACCOUNT_COUNT
// throwaway accounts instead, with random passwords that never leave the run.
//
// Passwords, cookies and tokens are never logged.
import { sleep } from "k6";
import encoding from "k6/encoding";
import http from "k6/http";

import { getEnvironmentConfig } from "../configs/environment.js";

const config = getEnvironmentConfig();
const AUTH_URL = `${config.FRONTEND_URL}/api/auth`;

// Better Auth lets an IP make 3 sign-in (or sign-up) requests, then refuses
// more until 10 s pass without one, so setup() sends them in bursts of 3.
const AUTH_BURST = 3;
const AUTH_PAUSE_SECONDS = 11;
const MAX_ATTEMPTS = 5;

const DEFAULT_THROWAWAY_ACCOUNTS = 5;
const MAX_THROWAWAY_ACCOUNTS = 100;
// Previews allow sign-up for this domain, and nothing is mailed to it.
const THROWAWAY_EMAIL_DOMAIN = "previews.agpt.co";

// A VU refreshes its token 10 to 20 minutes before the token expires, at a
// random point so the VUs sharing an account don't all refresh at once.
const REFRESH_MARGIN_MS = 10 * 60 * 1000;
const REFRESH_SPREAD_MS = 10 * 60 * 1000;
const REFRESH_RETRY_MS = 60 * 1000;

const ACCOUNTS_FORMAT = 'a JSON array of {"email": "...", "password": "..."}';

// Each VU runs this module in its own runtime, so this is per-VU state.
let vuSession = null;

// For options.setupTimeout: setup() pauses between bursts of sign-ins.
export function authSetupTimeout() {
  const listed = readAccounts();
  const count = listed ? listed.length : readAccountCount();
  const pauses = Math.ceil(count / AUTH_BURST) - 1;
  return `${60 + pauses * AUTH_PAUSE_SECONDS + count * 10}s`;
}

// Call from setup() and return the result (merged with any other setup data).
export function setupAuth() {
  const listed = readAccounts();
  if (!listed && !config.SIGNUP_ALLOWED) {
    throw new Error(
      `${config.NAME} needs LOAD_TEST_ACCOUNTS, ${ACCOUNTS_FORMAT}. See README.md.`,
    );
  }

  const accounts = listed || throwawayAccounts(readAccountCount());
  const sessions = accounts.map((account, index) => {
    if (index > 0 && index % AUTH_BURST === 0) sleep(AUTH_PAUSE_SECONDS);
    const label = `account ${index + 1} of ${accounts.length}`;
    return listed ? signIn(account, label) : signUp(account, index, label);
  });

  checkBackendAcceptsToken(sessions[0].token);
  console.log(
    `${listed ? "Signed in" : "Signed up"} ${sessions.length} account(s) at ${config.FRONTEND_URL}`,
  );
  return { auth: sessions };
}

// Headers for the current VU's account, from setupAuth()'s data.
export function authHeaders(data) {
  if (vuSession === null) {
    const sessions = data && data.auth;
    if (!sessions || sessions.length === 0) {
      throw new Error(
        "No sessions: the test's setup() must return setupAuth()'s result",
      );
    }
    const session = sessions[(__VU - 1) % sessions.length];
    vuSession = {
      cookies: session.cookies,
      token: session.token,
      refreshAt: refreshTime(session.expiresAt),
    };
  }
  if (Date.now() >= vuSession.refreshAt) refreshToken();
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${vuSession.token}`,
  };
}

function readAccounts() {
  const raw = __ENV.LOAD_TEST_ACCOUNTS;
  if (!raw) return null;

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    // The parser's own message can quote the input, passwords included.
    throw new Error(
      `LOAD_TEST_ACCOUNTS is not valid JSON: use ${ACCOUNTS_FORMAT}`,
    );
  }
  const valid =
    Array.isArray(parsed) &&
    parsed.length > 0 &&
    parsed.every(
      (account) =>
        account &&
        isNonEmptyString(account.email) &&
        isNonEmptyString(account.password),
    );
  if (!valid) {
    throw new Error(`LOAD_TEST_ACCOUNTS must be ${ACCOUNTS_FORMAT}`);
  }
  return parsed.map(({ email, password }) => ({ email, password }));
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.length > 0;
}

function readAccountCount() {
  const raw = __ENV.LOAD_TEST_ACCOUNT_COUNT;
  if (raw === undefined || raw === "") return DEFAULT_THROWAWAY_ACCOUNTS;
  const count = Number(raw);
  if (!Number.isInteger(count) || count < 1 || count > MAX_THROWAWAY_ACCOUNTS) {
    throw new Error(
      `LOAD_TEST_ACCOUNT_COUNT must be a whole number from 1 to ${MAX_THROWAWAY_ACCOUNTS}`,
    );
  }
  return count;
}

function throwawayAccounts(count) {
  const runId = randomHex(4);
  const accounts = [];
  for (let i = 1; i <= count; i++) {
    accounts.push({
      email: `loadtest-${runId}-${i}@${THROWAWAY_EMAIL_DOMAIN}`,
      password: randomHex(24),
    });
  }
  return accounts;
}

function randomHex(bytes) {
  const buffer = new Uint8Array(bytes);
  crypto.getRandomValues(buffer);
  return Array.from(buffer, (b) => b.toString(16).padStart(2, "0")).join("");
}

function signIn(account, label) {
  const jar = new http.CookieJar();
  const res = authPost(
    "/sign-in/email",
    { email: account.email, password: account.password },
    jar,
  );
  if (res.status !== 200) {
    throw new Error(
      `Signing in ${label} from LOAD_TEST_ACCOUNTS failed: ${describe(res)}`,
    );
  }
  return startSession(jar, label, "");
}

function signUp(account, index, label) {
  const jar = new http.CookieJar();
  const res = authPost(
    "/sign-up/email",
    {
      name: `Load test ${index + 1}`,
      email: account.email,
      password: account.password,
    },
    jar,
  );
  if (res.status !== 200) {
    throw new Error(
      `Signing up throwaway ${label} at ${config.FRONTEND_URL} failed: ${describe(res)}. ` +
        "Set LOAD_TEST_ACCOUNTS to use existing accounts instead.",
    );
  }
  return startSession(
    jar,
    label,
    " A sign-up gets no session when the target requires email verification;" +
      " set LOAD_TEST_ACCOUNTS to verified accounts instead.",
  );
}

function startSession(jar, label, hint) {
  const cookies = {};
  const byName = jar.cookiesForURL(`${AUTH_URL}/token`);
  for (const name of Object.keys(byName)) cookies[name] = byName[name][0];

  const res = requestToken(cookies);
  if (res.status !== 200) {
    throw new Error(
      `Getting a token for ${label} failed: ${describe(res)}.${hint}`,
    );
  }
  const token = readToken(res);
  return { cookies, token, expiresAt: tokenExpiry(token) };
}

// If the backend rejects the token, or is down or elsewhere, every request
// in the run fails fast, which would otherwise look like a very fast run.
function checkBackendAcceptsToken(token) {
  const res = http.get(`${config.API_BASE_URL}/credits`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401 || res.status === 403) {
    throw new Error(
      `${config.API_BASE_URL} rejected the token from ${config.FRONTEND_URL} ` +
        `(${describe(res)}): the backend must verify tokens against that frontend's /api/auth/jwks`,
    );
  }
  if (res.status !== 200) {
    throw new Error(
      `The first authenticated request, to ${config.API_BASE_URL}/credits, failed ` +
        `(${describe(res)}): check that the backend is up and that K6_ENVIRONMENT points at it`,
    );
  }
}

function refreshToken() {
  const res = requestToken(vuSession.cookies);
  if (res.status === 200) {
    try {
      const token = readToken(res);
      vuSession.token = token;
      vuSession.refreshAt = refreshTime(tokenExpiry(token));
      return;
    } catch (e) {
      // Fall through and retry later, like any other failed refresh.
    }
  }
  console.warn(
    `VU ${__VU} could not refresh its token (HTTP ${res.status}); retrying in a minute`,
  );
  vuSession.refreshAt = Date.now() + REFRESH_RETRY_MS;
}

function refreshTime(expiresAt) {
  return expiresAt - REFRESH_MARGIN_MS - Math.random() * REFRESH_SPREAD_MS;
}

function authPost(path, body, jar) {
  return withRetries(() =>
    http.post(`${AUTH_URL}${path}`, JSON.stringify(body), {
      headers: { "Content-Type": "application/json" },
      jar,
      redirects: 0,
    }),
  );
}

// The session cookies go in a fresh jar each time, so a VU's own jar never
// collects them and they reach only the frontend.
function requestToken(cookies) {
  return withRetries(() =>
    http.get(`${AUTH_URL}/token`, {
      cookies,
      jar: new http.CookieJar(),
      redirects: 0,
    }),
  );
}

function withRetries(send) {
  let res = send();
  for (
    let attempt = 1;
    res.status === 429 && attempt < MAX_ATTEMPTS;
    attempt++
  ) {
    sleep(retryAfterSeconds(res) + 1);
    res = send();
  }
  return res;
}

function retryAfterSeconds(res) {
  const seconds = parseInt(res.headers["X-Retry-After"], 10);
  return Number.isFinite(seconds) && seconds > 0
    ? Math.min(seconds, 60)
    : AUTH_PAUSE_SECONDS;
}

function readToken(res) {
  let token;
  try {
    token = res.json("token");
  } catch (e) {
    token = undefined;
  }
  if (typeof token !== "string" || token.split(".").length !== 3) {
    throw new Error(`${AUTH_URL}/token answered without a JWT`);
  }
  return token;
}

function tokenExpiry(token) {
  const payload = JSON.parse(
    encoding.b64decode(token.split(".")[1], "rawurl", "s"),
  );
  if (typeof payload.exp !== "number") {
    throw new Error(`The JWT from ${AUTH_URL}/token has no exp claim`);
  }
  return payload.exp * 1000;
}

// Error bodies carry a code and a message (Better Auth) or a detail (the
// backend). None of them repeats the credentials that were sent.
function describe(res) {
  if (res.status === 0) return res.error || "no response";
  let detail = "";
  try {
    const body = res.json();
    detail = (body && (body.code || body.message || body.detail)) || "";
  } catch (e) {
    detail = "";
  }
  if (typeof detail !== "string") detail = "";
  return `HTTP ${res.status}${detail ? ` ${String(detail).slice(0, 160)}` : ""}`;
}
