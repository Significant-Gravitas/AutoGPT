import type { APIRequestContext } from "@playwright/test";

const DEFAULT_EMAIL = "perf.harness@example.com";
const DEFAULT_PASSWORD = "perf-harness-password";
/** First-visit cards that would cover the page the scenario measures. */
const DONE_ONBOARDING_STEPS = [
  "ONBOARDING_COMPLETE",
  "AGENTS_TAB_INTRO",
  "MARKETPLACE_TAB_INTRO",
  "BUILD_TAB_INTRO",
];

interface Credentials {
  email: string;
  password: string;
}

function credentials(): Credentials {
  return {
    email: process.env.PERF_USER_EMAIL || DEFAULT_EMAIL,
    password: process.env.PERF_USER_PASSWORD || DEFAULT_PASSWORD,
  };
}

async function signInRequest(
  request: APIRequestContext,
  baseURL: string,
  { email, password }: Credentials,
) {
  return request.post(`${baseURL}/api/auth/sign-in/email`, {
    data: { email, password },
    headers: { Origin: baseURL },
  });
}

/**
 * Signs the browser context in through Better Auth, creating the perf user
 * the first time. With `--storage-state` (e.g. a `.auth/*.json` written by
 * the Playwright global setup) the context is already signed in and this
 * only reads the user id.
 */
export async function signIn(
  request: APIRequestContext,
  baseURL: string,
  hasStorageState: boolean,
) {
  if (!hasStorageState) {
    const creds = credentials();
    let response = await signInRequest(request, baseURL, creds);
    if (!response.ok()) {
      await request.post(`${baseURL}/api/auth/sign-up/email`, {
        data: { ...creds, name: "Perf Harness" },
        headers: { Origin: baseURL },
      });
      response = await signInRequest(request, baseURL, creds);
    }
    if (!response.ok()) {
      throw new Error(
        `Could not sign in as ${creds.email} (${response.status()}). Set PERF_USER_EMAIL / PERF_USER_PASSWORD to an account that exists, or pass --storage-state.`,
      );
    }
  }
  const session = await request.get(`${baseURL}/api/auth/get-session`);
  const body = (await session.json().catch(() => null)) as {
    user?: { id?: string };
  } | null;
  const userID = body?.user?.id;
  if (!userID) {
    throw new Error("Signed in, but /api/auth/get-session returned no user.");
  }
  for (const step of DONE_ONBOARDING_STEPS) {
    await request.post(`${baseURL}/api/proxy/api/onboarding/step?step=${step}`);
  }
  return userID;
}
