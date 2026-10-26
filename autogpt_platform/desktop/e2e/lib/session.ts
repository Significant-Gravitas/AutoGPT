// What the tests do inside the app's window. Requests are made by the page
// itself (fetch in the window), so they carry the window's own cookies and
// origin, exactly as the app's own requests do.
//
// The form selectors are the ones upstream's own browser tests use
// (frontend/src/playwright/utils/signup.ts and pages/login.page.ts) for the
// forms in frontend/src/app/(no-navbar)/signup and login.

import type { Page } from "@playwright/test";

const NAVIGATION_TIMEOUT_MS = 60_000;
const SIGNED_IN = /^\/(onboarding|marketplace|copilot|home|library)/;

export interface Reply {
  status: number;
  /** The parsed JSON body, or the text when it is not JSON. */
  body: any;
}

export async function call(page: Page, method: string, path: string, body?: unknown): Promise<Reply> {
  const reply = await settled(page, () =>
    page.evaluate(
      async ({ method, path, payload }) => {
        const response = await fetch(path, {
          method,
          headers: payload === null ? {} : { "content-type": "application/json" },
          body: payload,
        });
        return { status: response.status, text: await response.text() };
      },
      { method, path, payload: body === undefined ? null : JSON.stringify(body) },
    ),
  );
  try {
    return { status: reply.status, body: JSON.parse(reply.text) };
  } catch {
    return { status: reply.status, body: reply.text };
  }
}

/** The app redirects after signing in (to onboarding, then onwards). A
 * request made while the window is changing page dies with the old page;
 * make it again once the new one has loaded. */
async function settled<T>(page: Page, action: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await action();
    } catch (error) {
      const navigated = /context was destroyed|navigation/i.test(String(error));
      if (!navigated || attempt === 3) throw error;
      await page.waitForLoadState("load");
    }
  }
}

export async function signUp(page: Page, url: string, email: string, password: string): Promise<void> {
  await page.goto(`${url}/signup`);
  await page.getByText("Create your account").waitFor({ timeout: NAVIGATION_TIMEOUT_MS });
  await page.getByLabel("Email", { exact: true }).fill(email);
  await page.locator("#password").fill(password);
  await page.locator("#confirmPassword").fill(password);
  await page.getByRole("checkbox", { name: /agree to the terms/i }).click();
  await page.getByRole("button", { name: "Sign up" }).click();
  await page.waitForURL((target) => SIGNED_IN.test(target.pathname), { timeout: NAVIGATION_TIMEOUT_MS });
}

export async function signIn(page: Page, url: string, email: string, password: string): Promise<void> {
  await page.goto(`${url}/login`);
  await page.getByText("Log in to your account to continue").waitFor({ timeout: NAVIGATION_TIMEOUT_MS });
  await page.getByLabel("Email", { exact: true }).fill(email);
  await page.locator('input[type="password"]').fill(password);
  await page.getByRole("button", { name: "Log in", exact: true }).click();
  await page.waitForURL((target) => SIGNED_IN.test(target.pathname), { timeout: NAVIGATION_TIMEOUT_MS });
}

/** End the session where it is kept, then leave the signed-in pages. */
export async function signOut(page: Page, url: string): Promise<void> {
  const reply = await call(page, "POST", "/api/auth/sign-out", {});
  if (reply.status !== 200) throw new Error(`sign-out returned ${reply.status}: ${JSON.stringify(reply.body)}`);
  await page.goto(`${url}/login`);
}

/** The signed-in user as Better Auth reports it, or null when signed out. */
export async function currentUser(page: Page): Promise<{ email: string; role: string } | null> {
  const reply = await call(page, "GET", "/api/auth/get-session");
  return reply.body && typeof reply.body === "object" ? (reply.body.user ?? null) : null;
}

/** Skip the first-run questionnaire, as upstream's tests do
 * (frontend/src/playwright/utils/onboarding.ts). */
export async function completeOnboarding(page: Page): Promise<Reply> {
  return call(page, "POST", "/api/proxy/api/onboarding/step?step=ONBOARDING_COMPLETE");
}

/** Electron puts "<app name>/<app version>" in the user agent, ahead of Chrome's. */
export async function appVersion(page: Page): Promise<string> {
  const agent = await page.evaluate(() => navigator.userAgent);
  const match = /\s[^\s/]+\/(\d+\.\d+\.\d+\S*) Chrome\//.exec(agent);
  if (!match) throw new Error(`the user agent does not name the app's version: ${agent}`);
  return match[1];
}

export function isNewer(version: string, than: string): boolean {
  const parts = (value: string) => value.split("-")[0].split(".").map(Number);
  const [a, b] = [parts(version), parts(than)];
  for (let index = 0; index < 3; index += 1) {
    if (a[index] !== b[index]) return a[index] > b[index];
  }
  return false;
}
