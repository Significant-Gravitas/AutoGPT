import fs from "fs";
import path from "path";
import { LoginPage } from "../pages/login.page";
import {
  SEEDED_AUTH_STATE_ACCOUNT_KEYS,
  SEEDED_TEST_ACCOUNTS,
  SEEDED_TEST_USERS,
  getAuthStatePath,
} from "../credentials/accounts";
import { buildCookieConsentStorageState } from "../credentials/storage-state";
import { signupTestUser } from "./signup";
import { getBrowser } from "./get-browser";
import { skipOnboardingIfPresent } from "./onboarding";

export interface TestUser {
  email: string;
  password: string;
  id?: string;
  createdAt?: string;
}

export interface UserPool {
  users: TestUser[];
  createdAt: string;
  version: string;
}

const AUTH_STATE_KEYS = [...SEEDED_AUTH_STATE_ACCOUNT_KEYS];

export async function createTestUser(
  email?: string,
  password?: string,
  ignoreOnboarding: boolean = true,
): Promise<TestUser> {
  const { faker } = await import("@faker-js/faker");
  const userEmail = email || faker.internet.email();
  const userPassword = password || faker.internet.password({ length: 12 });

  try {
    const browser = await getBrowser();
    const context = await browser.newContext();
    const page = await context.newPage();

    // Auto-accept cookies in test environment to prevent banner from appearing
    await page.addInitScript(() => {
      window.localStorage.setItem(
        "autogpt_cookie_consent",
        JSON.stringify({
          hasConsented: true,
          timestamp: Date.now(),
          analytics: true,
          monitoring: true,
        }),
      );
    });

    try {
      const testUser = await signupTestUser(
        page,
        userEmail,
        userPassword,
        ignoreOnboarding,
        false,
      );
      return testUser;
    } finally {
      await page.close();
      await context.close();
      await browser.close();
    }
  } catch (error) {
    console.error(`❌ Error creating test user ${userEmail}:`, error);
    throw error;
  }
}

export async function createTestUsers(count: number): Promise<TestUser[]> {
  console.log(`👥 Creating ${count} test users...`);

  const users: TestUser[] = [];
  let consecutiveFailures = 0;

  for (let i = 0; i < count; i++) {
    try {
      const user = await createTestUser();
      users.push(user);
      consecutiveFailures = 0; // Reset failure counter on success
      console.log(`✅ Created user ${i + 1}/${count}: ${user.email}`);

      // Small delay to prevent overwhelming the system
      if (i < count - 1) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    } catch (error) {
      consecutiveFailures++;
      console.error(`❌ Failed to create user ${i + 1}/${count}:`, error);

      // If we have too many consecutive failures, stop trying
      if (consecutiveFailures >= 3) {
        console.error(
          `⚠️ Stopping after ${consecutiveFailures} consecutive failures`,
        );
        break;
      }

      // Add a longer delay after failure to let system recover
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  console.log(`🎉 Successfully created ${users.length}/${count} test users`);
  return users;
}

export async function getTestUser(): Promise<TestUser> {
  if (SEEDED_TEST_USERS.length === 0) {
    throw new Error("No seeded E2E users are configured");
  }

  const randomIndex = Math.floor(Math.random() * SEEDED_TEST_USERS.length);
  const { email, password } = SEEDED_TEST_USERS[randomIndex];
  return { email, password };
}

function hasStoredAuthState(accountKey: (typeof AUTH_STATE_KEYS)[number]) {
  return fs.existsSync(getAuthStatePath(accountKey));
}

export function hasSeededAuthStates(): boolean {
  return AUTH_STATE_KEYS.every((accountKey) => hasStoredAuthState(accountKey));
}

async function createAuthStateForUser(
  baseURL: string,
  accountKey: (typeof AUTH_STATE_KEYS)[number],
): Promise<void> {
  const browser = await getBrowser();

  try {
    const { email, password } = SEEDED_TEST_ACCOUNTS[accountKey];
    const origin = new URL(baseURL).origin;
    const context = await browser.newContext({
      baseURL,
      storageState: buildCookieConsentStorageState(origin),
    });
    const page = await context.newPage();
    const loginPage = new LoginPage(page);

    await page.goto("/login");
    await loginPage.login(email, password);
    await page.waitForURL(
      (url: URL) =>
        /\/(onboarding|marketplace|copilot|library)/.test(url.pathname),
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
  } catch (error) {
    const { email } = SEEDED_TEST_ACCOUNTS[accountKey];
    throw new Error(
      `Failed to create auth state for ${email}. Seed the backend test data with backend/test/e2e_test_data.py before running Playwright. ${String(
        error,
      )}`,
    );
  } finally {
    await browser.close();
  }
}

export async function ensureSeededAuthStates(baseURL: string): Promise<void> {
  for (const accountKey of AUTH_STATE_KEYS) {
    if (hasStoredAuthState(accountKey)) {
      continue;
    }

    await createAuthStateForUser(baseURL, accountKey);
  }
}
