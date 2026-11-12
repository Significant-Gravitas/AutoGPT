import type { Browser, Page } from "@playwright/test";
import { expect, test } from "../coverage-fixture";
import { JOURNEY_USER_AGENT } from "./user-agent";

// Several people signing up to one self-hosted install at once. That burst
// once ran the appliance's frontend out of database connections: sign-ups
// failed, and signed-in requests reached the backend without a token. Every
// sign-up here must land in onboarding, and every signed-in API call must
// answer 200, neither a 401 (sent without a token) nor a 503 (no token could
// be minted).
//
// Run it against a self-hosted build, e.g. the single-container appliance:
//   PLAYWRIGHT_BASE_URL=http://localhost:3400 \
//   PLAYWRIGHT_SELFHOST_EMAIL_DOMAIN=patrol.example.com \
//     pnpm exec playwright test --config playwright.journeys.config.ts \
//       --project=selfhost signup-burst
// PLAYWRIGHT_SIGNUP_BURST_USERS sets how many sign up at once (default 6).

const EMAIL_DOMAIN =
  process.env.PLAYWRIGHT_SELFHOST_EMAIL_DOMAIN ?? "patrol.example.com";
const USERS = Number(process.env.PLAYWRIGHT_SIGNUP_BURST_USERS ?? 6);
const REQUESTS_PER_USER = 30;

test.setTimeout(240_000);

test("signup burst: everyone signing up at once gets in, and every signed-in request is authenticated", async ({
  browser,
}, testInfo) => {
  const runID = Date.now();
  const baseURL = String(testInfo.project.use.baseURL);
  const problems: string[] = [];

  const signUpWave = (wave: string, count: number) =>
    Promise.all(
      Array.from({ length: count }, (_, i) =>
        signUp(browser, baseURL, `burst-${runID}-${wave}${i}@${EMAIL_DOMAIN}`),
      ),
    );

  // The second wave signs up while the first one is calling the API, so
  // sign-ups and token mints compete for connections the way they do when
  // a team starts using a fresh install together.
  const firstWave = await signUpWave("a", Math.ceil(USERS / 2));
  const [secondWave] = await Promise.all([
    signUpWave("b", Math.floor(USERS / 2)),
    ...firstWave.map((user) => callTheApi(user, problems)),
  ]);
  await Promise.all(secondWave.map((user) => callTheApi(user, problems)));

  const users = [...firstWave, ...secondWave];
  const signUpFailures = users.flatMap((user) => user.problems);
  await Promise.all(users.map((user) => user.page.context().close()));

  expect(signUpFailures, "every sign-up lands in onboarding").toEqual([]);
  expect(problems, "every signed-in API call answers 200").toEqual([]);
});

interface BurstUser {
  email: string;
  page: Page;
  problems: string[];
}

async function signUp(
  browser: Browser,
  baseURL: string,
  email: string,
): Promise<BurstUser> {
  const context = await browser.newContext({
    baseURL,
    userAgent: JOURNEY_USER_AGENT,
  });
  const page = await context.newPage();
  const problems: string[] = [];
  const password = `Burst-${Date.now()}-journey`;

  await page.goto("/signup");
  await expect(page.getByText("Create your account")).toBeVisible();
  await page.getByLabel("Email").fill(email);
  await page.locator("#password").fill(password);
  await page.locator("#confirmPassword").fill(password);
  await page.getByRole("button", { name: "Sign up" }).click();
  try {
    await page.waitForURL((url) => !url.pathname.startsWith("/signup"), {
      timeout: 60_000,
    });
  } catch {
    const shown = await page
      .getByRole("alert")
      .or(page.locator("[data-sonner-toast], [role=status]"))
      .allInnerTexts();
    problems.push(`${email}: sign-up did not complete (${shown.join(" | ")})`);
  }
  return { email, page, problems };
}

// A signed-in page asks the API for many things at once.
async function callTheApi(user: BurstUser, problems: string[]) {
  if (user.problems.length > 0) return;
  const statuses = await Promise.all(
    Array.from({ length: REQUESTS_PER_USER }, () =>
      user.page.request
        .get("/api/proxy/api/library/agents?page_size=1")
        .then((response) => response.status()),
    ),
  );
  for (const status of statuses.filter((status) => status !== 200)) {
    problems.push(`${user.email}: GET /api/library/agents answered ${status}`);
  }
}
