import fs from "fs";
import path from "path";
import { LoginPage } from "../pages/login.page";
import {
  SEEDED_AUTH_STATE_ACCOUNT_KEYS,
  SEEDED_TEST_ACCOUNTS,
  getAuthStatePath,
} from "../credentials/accounts";
import { getBrowser } from "./get-browser";
import { skipOnboardingIfPresent } from "./onboarding";

export interface TestUser {
  email: string;
  password: string;
  id?: string;
  createdAt?: string;
}

const AUTH_STATE_KEYS = [...SEEDED_AUTH_STATE_ACCOUNT_KEYS];

function hasStoredAuthState(accountKey: (typeof AUTH_STATE_KEYS)[number]) {
  return fs.existsSync(getAuthStatePath(accountKey));
}

function authStateMatchesOrigin(
  accountKey: (typeof AUTH_STATE_KEYS)[number],
  origin: string,
): boolean {
  const statePath = getAuthStatePath(accountKey);
  if (!fs.existsSync(statePath)) {
    return false;
  }

  try {
    const state = JSON.parse(fs.readFileSync(statePath, "utf8")) as {
      origins?: Array<{ origin?: string }>;
    };
    return (
      state.origins?.some((storedOrigin) => storedOrigin.origin === origin) ??
      false
    );
  } catch {
    return false;
  }
}

async function authStateHasLiveSession(
  baseURL: string,
  accountKey: (typeof AUTH_STATE_KEYS)[number],
): Promise<boolean> {
  const browser = await getBrowser();

  try {
    const context = await browser.newContext({
      baseURL,
      storageState: getAuthStatePath(accountKey),
    });
    const page = await context.newPage();

    try {
      await page.goto("/marketplace");
      await page.waitForLoadState("domcontentloaded");
      await skipOnboardingIfPresent(page, "/marketplace");
      return await page
        .getByTestId("profile-popout-menu-trigger")
        .waitFor({ state: "visible", timeout: 10_000 })
        .then(() => true)
        .catch(() => false);
    } finally {
      await page.close();
      await context.close();
    }
  } catch {
    return false;
  } finally {
    await browser.close();
  }
}

export async function getInvalidSeededAuthStateKeys(
  baseURL: string,
): Promise<(typeof AUTH_STATE_KEYS)[number][]> {
  const origin = new URL(baseURL).origin;
  const invalidKeys = await Promise.all(
    AUTH_STATE_KEYS.map(async (accountKey) => {
      if (
        !hasStoredAuthState(accountKey) ||
        !authStateMatchesOrigin(accountKey, origin)
      ) {
        return accountKey;
      }

      return (await authStateHasLiveSession(baseURL, accountKey))
        ? null
        : accountKey;
    }),
  );

  return invalidKeys.filter(
    (accountKey): accountKey is (typeof AUTH_STATE_KEYS)[number] =>
      accountKey !== null,
  );
}

// Seed logins are retried because the /login form is gated behind client
// hydration + a getCurrentUser() round-trip; on constrained CI runners a single
// attempt occasionally exceeds the form timeout. Without a retry one slow login
// fails the whole suite, since global setup is not covered by Playwright's
// `retries`.
const AUTH_SETUP_MAX_ATTEMPTS = 3;

// Cap how many seed logins run at once. Logging in every account in parallel
// overwhelms the frontend + auth service on a 2-vCPU runner, which is what
// pushed the form render past its timeout to begin with.
const AUTH_SETUP_CONCURRENCY = 2;

async function createAuthStateForUser(
  baseURL: string,
  accountKey: (typeof AUTH_STATE_KEYS)[number],
): Promise<void> {
  const { email } = SEEDED_TEST_ACCOUNTS[accountKey];

  for (let attempt = 1; attempt <= AUTH_SETUP_MAX_ATTEMPTS; attempt += 1) {
    try {
      await attemptCreateAuthState(baseURL, accountKey);
      return;
    } catch (error) {
      if (attempt === AUTH_SETUP_MAX_ATTEMPTS) {
        throw new Error(
          `Failed to create auth state for ${email} after ${AUTH_SETUP_MAX_ATTEMPTS} attempts: ${String(
            error,
          )}. If these seeded QA accounts are missing, seed them with backend/test/e2e_test_data.py before running Playwright.`,
        );
      }
      console.warn(
        `⚠️ Auth state seeding for ${email} failed on attempt ${attempt}/${AUTH_SETUP_MAX_ATTEMPTS}, retrying: ${String(
          error,
        )}`,
      );
    }
  }
}

async function attemptCreateAuthState(
  baseURL: string,
  accountKey: (typeof AUTH_STATE_KEYS)[number],
): Promise<void> {
  const browser = await getBrowser();

  try {
    const { email, password } = SEEDED_TEST_ACCOUNTS[accountKey];
    const context = await browser.newContext({ baseURL });
    const page = await context.newPage();
    const loginPage = new LoginPage(page);

    await page.goto("/login");
    await loginPage.login(email, password);
    await page.waitForURL(
      (url: URL) =>
        /\/(onboarding|marketplace|copilot|home|library)/.test(url.pathname),
      { timeout: 20000 },
    );
    await skipOnboardingIfPresent(page, "/marketplace");
    await page.getByTestId("profile-popout-menu-trigger").waitFor({
      state: "visible",
      timeout: 10000,
    });

    const statePath = getAuthStatePath(accountKey);
    fs.mkdirSync(path.dirname(statePath), { recursive: true });
    await context.storageState({ path: statePath });
    await context.close();
  } finally {
    await browser.close();
  }
}

export async function ensureSeededAuthStates(baseURL: string): Promise<void> {
  const invalidKeys = await getInvalidSeededAuthStateKeys(baseURL);

  const results = await mapWithConcurrency(
    invalidKeys,
    AUTH_SETUP_CONCURRENCY,
    (accountKey) => createAuthStateForUser(baseURL, accountKey),
  );

  const firstFailure = results.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  );
  if (firstFailure) {
    throw firstFailure.reason;
  }
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  task: (item: T) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let cursor = 0;

  async function worker(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      try {
        results[index] = {
          status: "fulfilled",
          value: await task(items[index]),
        };
      } catch (reason) {
        results[index] = { status: "rejected", reason };
      }
    }
  }

  const workerCount = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}
