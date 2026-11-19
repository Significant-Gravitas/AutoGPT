// Better Auth sign-in for the load tests.
//
// The backend only accepts the ES256 JWTs that the frontend's Better Auth
// issues at GET /api/auth/token (audience "authenticated", valid for an hour).
// setup() signs each account in once per run and gets its token; every VU
// then uses those tokens. A run therefore signs in once per account, however
// many VUs and iterations it has.
//
// Accounts come from LOAD_TEST_ACCOUNTS, a JSON array of
// {"email": "...", "password": "..."}. When that is unset and the target
// allows it (PREVIEW and LOCAL), setup() signs up LOAD_TEST_ACCOUNT_COUNT
// throwaway accounts instead, with random passwords that never leave the run.
//
// Each account has a mode, the copilot engine its turns run on, and its
// address names it. A target with CHAT_LOAD_TEST_EMAIL_DOMAIN set to the
// accounts' domain (previews.agpt.co on a preview) routes
// loadtest-dummy-…@<domain> to the dummy engine and loadtest-scripted-…@
// <domain> to the SDK engine against the scripted model; every other address
// gets the real model. The mode is read from the address the same way, so an
// account in LOAD_TEST_ACCOUNTS gets the mode its address asks for; the
// setup probes (lib/seed.js) confirm the target agrees before any turn
// depends on it.
//
// Passwords, cookies and tokens are never logged.
import { sleep } from "k6";
import encoding from "k6/encoding";
import http from "k6/http";

import { getEnvironmentConfig } from "../configs/environment.js";
import { enabledJourneys } from "../configs/profiles.js";

const config = getEnvironmentConfig();
const AUTH_URL = `${config.FRONTEND_URL}/api/auth`;

// Better Auth lets an IP make 3 sign-in (or sign-up) requests, then refuses
// more until 10 s pass without one, so setup() sends them in bursts of 3.
const AUTH_BURST = 3;
const AUTH_PAUSE_SECONDS = 11;
const MAX_ATTEMPTS = 5;

// Copilot caps running turns per user (5) and turns in flight, running or
// queued (15), so turns are spread over enough accounts that the load, not
// the cap, sets the pace.
const DEFAULT_THROWAWAY_ACCOUNTS = 8;
const MAX_THROWAWAY_ACCOUNTS = 100;
// Of the throwaway accounts, how many are scripted and how many real; the
// rest are dummy accounts, at least one.
const DEFAULT_SCRIPTED_ACCOUNTS = 2;
// Previews allow sign-up for this domain, and nothing is mailed to it.
const THROWAWAY_EMAIL_DOMAIN = "previews.agpt.co";

// The local part's prefix of each mode's addresses, matched as the backend
// matches them (backend/copilot/load_test.py); any other address is real.
const MODE_PREFIXES = {
  dummy: "loadtest-dummy-",
  scripted: "loadtest-scripted-",
};

// A VU refreshes a token 10 to 20 minutes before the token expires, at a
// random point so the VUs sharing an account don't all refresh at once.
const REFRESH_MARGIN_MS = 10 * 60 * 1000;
const REFRESH_SPREAD_MS = 10 * 60 * 1000;
const REFRESH_RETRY_MS = 60 * 1000;
const ACCOUNT_TOKEN_MARGIN_MS = 5 * 60 * 1000;

const ACCOUNTS_FORMAT = 'a JSON array of {"email": "...", "password": "..."}';

// Each VU runs this module in its own runtime, so this is per-VU state: the
// token of each account the VU has used, by the account's index.
const vuSessions = {};

// For options.setupTimeout: setup() pauses between bursts of sign-ins.
export function authSetupTimeout() {
  const listed = readAccounts();
  const count = listed ? listed.length : readAccountCount();
  const pauses = Math.ceil(count / AUTH_BURST) - 1;
  return `${60 + pauses * AUTH_PAUSE_SECONDS + count * 10}s`;
}

// Call from setup() and return the result (merged with any other setup data).
// Each session carries its account's mode; `signedUp` says the run created
// the accounts, so everything under them is the run's. `options.mode` makes
// every account that mode, and refuses accounts in LOAD_TEST_ACCOUNTS whose
// address asks for another.
export function setupAuth(options) {
  const only = (options && options.mode) || null;
  const listed = readAccounts();
  if (!listed && !config.SIGNUP_ALLOWED) {
    throw new Error(
      `${config.NAME} needs LOAD_TEST_ACCOUNTS, ${ACCOUNTS_FORMAT}. See README.md.`,
    );
  }
  if (listed && only) {
    listed.forEach((account, index) => {
      const mode = modeForEmail(account.email);
      if (mode !== only) {
        throw new Error(
          `Account ${index + 1} in LOAD_TEST_ACCOUNTS is named for the ${mode} engine; this test needs ${only} accounts (README: Account modes)`,
        );
      }
    });
  }

  const defaultCount = options && options.defaultCount;
  const accounts =
    listed || throwawayAccounts(readAccountCount(defaultCount), only);
  const sessions = accounts.map((account, index) => {
    if (index > 0 && index % AUTH_BURST === 0) sleep(AUTH_PAUSE_SECONDS);
    const label = `account ${index + 1} of ${accounts.length}`;
    const session = listed
      ? signIn(account, label)
      : signUp(account, index, label);
    session.mode = modeForEmail(account.email);
    return session;
  });

  checkBackendAcceptsToken(sessions[0].token);
  const modes = countModes(sessions.map((session) => session.mode));
  console.log(
    `${listed ? "Signed in" : "Signed up"} ${sessions.length} account(s) at ${config.FRONTEND_URL}: ` +
      `${modes.dummy} dummy, ${modes.scripted} scripted, ${modes.real} real`,
  );
  return { auth: sessions, signedUp: !listed };
}

// How many accounts a run signs in, from LOAD_TEST_ACCOUNTS or
// LOAD_TEST_ACCOUNT_COUNT; usable in the init context.
export function accountCount() {
  const listed = readAccounts();
  return listed ? listed.length : readAccountCount();
}

// How many of the run's accounts have each mode; usable in the init context.
export function accountModeCounts() {
  const listed = readAccounts();
  if (listed) return countModes(listed.map((a) => modeForEmail(a.email)));
  return throwawayPlan(readAccountCount());
}

// The engine an address's turns run on: "dummy", "scripted" or "real".
export function modeForEmail(email) {
  const local = String(email).trim().toLowerCase().split("@")[0];
  for (const mode of Object.keys(MODE_PREFIXES)) {
    if (local.indexOf(MODE_PREFIXES[mode]) === 0) return mode;
  }
  return "real";
}

function countModes(modes) {
  const counts = { dummy: 0, scripted: 0, real: 0 };
  for (const mode of modes) counts[mode]++;
  return counts;
}

// Headers for one account by index, for setup() and teardown(), which run
// outside any VU. A token within five minutes of expiring is renewed with
// the account's session first, so the teardown of a long run gets through.
export function accountHeaders(data, index) {
  const sessions = data && data.auth;
  if (!sessions || sessions.length === 0) {
    throw new Error("No sessions: setupAuth() must run first");
  }
  const session = sessions[index % sessions.length];
  if (
    !session.renewFailed &&
    session.expiresAt - Date.now() < ACCOUNT_TOKEN_MARGIN_MS
  ) {
    renewAccountToken(session, `account ${(index % sessions.length) + 1}`);
  }
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${session.token}`,
  };
}

function renewAccountToken(session, label) {
  const res = requestToken(session.cookies);
  if (res.status === 200) {
    try {
      session.token = readToken(res);
      session.expiresAt = tokenExpiry(session.token);
      return;
    } catch (e) {
      // Reported below, like any other failed renewal.
    }
  }
  session.renewFailed = true;
  console.warn(
    `Renewing the token of ${label} failed (HTTP ${res.status}); its requests use the old one`,
  );
}

// Headers for one account inside a VU, from setupAuth()'s data: the account
// at `index`, or the VU's own account when it is left out. VUs are numbered
// from 1 and VU n's own account is n - 1, round-robin, so the read journeys
// spread over every account; setup() and teardown() run as VU 0 and use the
// first. The turn journeys name their account, one of their mode's.
export function authHeaders(data, index) {
  const sessions = data && data.auth;
  if (!sessions || sessions.length === 0) {
    throw new Error(
      "No sessions: the test's setup() must return setupAuth()'s result",
    );
  }
  const i =
    index === undefined || index === null
      ? Math.max(__VU - 1, 0) % sessions.length
      : index % sessions.length;
  let session = vuSessions[i];
  if (!session) {
    session = {
      cookies: sessions[i].cookies,
      token: sessions[i].token,
      refreshAt: refreshTime(sessions[i].expiresAt),
    };
    vuSessions[i] = session;
  }
  if (Date.now() >= session.refreshAt) refreshToken(session, i);
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${session.token}`,
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

function readAccountCount(fallback) {
  const raw = __ENV.LOAD_TEST_ACCOUNT_COUNT;
  if (raw === undefined || raw === "") {
    return fallback || DEFAULT_THROWAWAY_ACCOUNTS;
  }
  const count = Number(raw);
  if (!Number.isInteger(count) || count < 1 || count > MAX_THROWAWAY_ACCOUNTS) {
    throw new Error(
      `LOAD_TEST_ACCOUNT_COUNT must be a whole number from 1 to ${MAX_THROWAWAY_ACCOUNTS}`,
    );
  }
  return count;
}

// How many scripted and real throwaway accounts the run signs up; the rest
// of `count` are dummy accounts. LOAD_TEST_SCRIPTED_ACCOUNTS defaults to 2,
// and LOAD_TEST_REAL_ACCOUNTS to 1 when the run sends real-model turns (else
// 0); a default gives way when `count` leaves no room for it.
function throwawayPlan(count) {
  const realWanted = enabledJourneys().includes("copilot_turn_real");
  const real = modeCount(
    "LOAD_TEST_REAL_ACCOUNTS",
    Math.min(realWanted ? 1 : 0, count - 1),
    count,
  );
  const scripted = modeCount(
    "LOAD_TEST_SCRIPTED_ACCOUNTS",
    Math.max(0, Math.min(DEFAULT_SCRIPTED_ACCOUNTS, count - 1 - real)),
    count,
  );
  if (real + scripted >= count) {
    throw new Error(
      `LOAD_TEST_SCRIPTED_ACCOUNTS (${scripted}) and LOAD_TEST_REAL_ACCOUNTS (${real}) leave none of the ` +
        `${count} accounts (LOAD_TEST_ACCOUNT_COUNT) for the dummy engine, which every turn journey needs first`,
    );
  }
  return { dummy: count - real - scripted, scripted, real };
}

function modeCount(variable, fallback, count) {
  const raw = __ENV[variable];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0 || n >= count) {
    throw new Error(
      `${variable} must be a whole number from 0 to ${count - 1} (LOAD_TEST_ACCOUNT_COUNT less one), got "${raw}"`,
    );
  }
  return n;
}

// Accounts are named after the run (RUN_ID from k6.sh, else a random tag),
// so everything a run creates can be found under its own accounts, and
// after their mode: loadtest-dummy-<run>-<n>, loadtest-scripted-<run>-<n>
// and loadtest-<run>-<n>, numbered across the run. `only` makes them all
// one mode.
function throwawayAccounts(count, only) {
  const runTag = (__ENV.RUN_ID || randomHex(4))
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-");
  const plan = only
    ? Object.assign({ dummy: 0, scripted: 0, real: 0 }, { [only]: count })
    : throwawayPlan(count);
  const accounts = [];
  for (const mode of ["dummy", "scripted", "real"]) {
    for (let k = 0; k < plan[mode]; k++) {
      const n = accounts.length + 1;
      const local =
        mode === "real"
          ? `loadtest-${runTag}-${n}`
          : `${MODE_PREFIXES[mode]}${runTag}-${n}`;
      const email = `${local}@${THROWAWAY_EMAIL_DOMAIN}`;
      // A run id that reads as another mode's prefix would route a real
      // account to an engine it was not meant for.
      if (modeForEmail(email) !== mode) {
        throw new Error(
          `RUN_ID "${__ENV.RUN_ID}" makes a ${mode} account's address read as ${modeForEmail(email)}; ` +
            "start it with something other than dummy- or scripted-",
        );
      }
      accounts.push({ email, password: randomHex(24) });
    }
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

function refreshToken(session, index) {
  const res = requestToken(session.cookies);
  if (res.status === 200) {
    try {
      const token = readToken(res);
      session.token = token;
      session.refreshAt = refreshTime(tokenExpiry(token));
      return;
    } catch (e) {
      // Fall through and retry later, like any other failed refresh.
    }
  }
  console.warn(
    `VU ${__VU} could not refresh the token of account ${index + 1} (HTTP ${res.status}); retrying in a minute`,
  );
  session.refreshAt = Date.now() + REFRESH_RETRY_MS;
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
