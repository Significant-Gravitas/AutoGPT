import type { BrowserContext, Page, Request } from "@playwright/test";
import { expect, test } from "../coverage-fixture";
import { skipOnboardingIfPresent } from "../utils/onboarding";
import { signupTestUser } from "../utils/signup";

// Journey: a signed-in user stays signed in through the network trouble a
// real laptop sees (waking up offline, a server hiccup, a deploy landing while
// the tab is open), and is still signed out when the auth backend says the
// session is gone. Sentry: BUILDER-3J3, BUILDER-3QH (forced logouts) and
// BUILDER-7MC/5BH/59A (false onboarding error toast).

const ONBOARDING_ERROR = "Failed to initialize onboarding";

// The session check is the `validateSession` server action: a POST carrying a
// `next-action` header whose only argument is the current path.
function isSessionCheck(request: Request) {
  return (
    request.method() === "POST" &&
    Boolean(request.headers()["next-action"]) &&
    (request.postData() ?? "").startsWith('["/')
  );
}

function journeyEmail(baseURL: string | undefined) {
  const unique = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  // Previews only accept password sign-ups from the previews domain.
  const domain = baseURL?.includes(".vercel.app")
    ? "previews.agpt.co"
    : "example.com";
  return `session-journey-${unique}@${domain}`;
}

async function signUpIntoLibrary(page: Page, baseURL: string | undefined) {
  await signupTestUser(page, journeyEmail(baseURL), undefined, false);
  await skipOnboardingIfPresent(page, "/library");
  await expect(page).toHaveURL(/\/library/);
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
}

// The tab coming back into view is what revalidates the session. Fire it
// until the app has actually made the attempt we are interfering with (the
// store throttles revalidation to once every 2 s).
async function refocusUntil(page: Page, attempted: () => boolean) {
  await expect(async () => {
    await page.evaluate(() =>
      document.dispatchEvent(new Event("visibilitychange")),
    );
    expect(attempted()).toBe(true);
  }).toPass({ intervals: [250, 500, 1000, 2000], timeout: 15_000 });
}

// Every trip to the login page, including one the login page bounces straight
// back from because the cookie is still good: that bounce is the forced logout
// users see, and it throws away whatever they had open.
function recordLoginVisits(page: Page) {
  const visits: string[] = [];
  page.on("framenavigated", (frame) => {
    if (
      frame === page.mainFrame() &&
      new URL(frame.url()).pathname === "/login"
    )
      visits.push(frame.url());
  });
  return visits;
}

// The account menu only renders for a signed-in user, so its trigger is the
// app's own statement of who is logged in.
async function expectStillSignedIn(page: Page, loginVisits: string[]) {
  await expect(page).toHaveURL(/\/library/);
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  expect(loginVisits, "the user was sent to the login page").toEqual([]);
}

async function nextSuccessfulSessionCheck(page: Page) {
  await page.waitForResponse(
    (response) => isSessionCheck(response.request()) && response.ok(),
    { timeout: 20_000 },
  );
}

function recordOnboardingErrors(page: Page) {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error" && message.text().includes(ONBOARDING_ERROR))
      errors.push(message.text());
  });
  return errors;
}

async function openSecondTab(context: BrowserContext) {
  const tab = await context.newPage();
  await tab.goto("/library");
  await expect(tab.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  return tab;
}

test("session resilience journey: network trouble never signs the user out", async ({
  page,
  baseURL,
}) => {
  test.setTimeout(120_000);
  const onboardingErrors = recordOnboardingErrors(page);

  await test.step("a dropped onboarding request is retried without an error toast", async () => {
    await signupTestUser(page, journeyEmail(baseURL), undefined, false);
    await expect(page).toHaveURL(/\/onboarding/);

    let dropped = 0;
    await page.route(
      "**/api/proxy/api/internal/onboarding/completed*",
      (route) => {
        if (dropped > 0) return route.fallback();
        dropped += 1;
        return route.abort("failed");
      },
    );
    await page.reload();
    await expect.poll(() => dropped, { timeout: 15_000 }).toBe(1);

    // Registered only now, so the first page load's answer can't satisfy it.
    const retried = page.waitForResponse(
      (response) =>
        response.url().includes("/api/internal/onboarding/completed") &&
        response.ok(),
      { timeout: 15_000 },
    );
    // Soft, so a regression here still lets the session steps report too.
    expect
      .soft(
        await retried.catch(() => null),
        "the dropped onboarding request should be retried",
      )
      .not.toBeNull();
    await expect.soft(page.getByText(ONBOARDING_ERROR)).toHaveCount(0);
    expect.soft(onboardingErrors).toEqual([]);
    await page.unrouteAll({ behavior: "ignoreErrors" });

    await skipOnboardingIfPresent(page, "/library");
    await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  });

  const loginVisits = recordLoginVisits(page);

  await test.step("a session check lost to a dropped connection keeps the session and recovers when the network returns", async () => {
    // A laptop waking before its network is back: the check on refocus
    // dies with "Failed to fetch" / "Load failed" (BUILDER-3J3).
    let dropped = 0;
    await page.route("**/*", (route) => {
      if (!isSessionCheck(route.request())) return route.fallback();
      dropped += 1;
      return route.abort("internetdisconnected");
    });

    await refocusUntil(page, () => dropped > 0);
    const recovered = nextSuccessfulSessionCheck(page);
    await page.unrouteAll({ behavior: "ignoreErrors" });
    await page.evaluate(() => window.dispatchEvent(new Event("online")));
    await expectStillSignedIn(page, loginVisits);
    await recovered;
  });

  await test.step("a 503 from the server keeps the session and is retried", async () => {
    let unavailable = 0;
    await page.route("**/*", (route) => {
      if (!isSessionCheck(route.request())) return route.fallback();
      unavailable += 1;
      return route.fulfill({
        status: 503,
        contentType: "text/plain",
        body: "Service Unavailable",
      });
    });

    await refocusUntil(page, () => unavailable > 0);
    const recovered = nextSuccessfulSessionCheck(page);
    await page.unrouteAll({ behavior: "ignoreErrors" });
    await expectStillSignedIn(page, loginVisits);
    await recovered;
  });

  await test.step("a deploy landing while the tab is open reloads instead of signing out", async () => {
    let stale = 0;
    await page.route("**/*", (route) => {
      if (!isSessionCheck(route.request()) || stale > 0)
        return route.fallback();
      stale += 1;
      // What Next answers when this bundle's server action IDs are gone.
      return route.fulfill({
        status: 404,
        headers: { "x-nextjs-action-not-found": "1" },
        contentType: "text/plain",
        body: "Server action not found.",
      });
    });
    await page.evaluate(() => {
      (window as unknown as { __beforeDeploy?: boolean }).__beforeDeploy = true;
    });

    const reloaded = page.waitForEvent("load", { timeout: 20_000 });
    await refocusUntil(page, () => stale > 0);
    await reloaded;
    const survivedReload = await page.evaluate(
      () =>
        (window as unknown as { __beforeDeploy?: boolean }).__beforeDeploy ??
        false,
    );
    expect(survivedReload, "the page should reload onto the new deploy").toBe(
      false,
    );
    await page.unrouteAll({ behavior: "ignoreErrors" });

    await expectStillSignedIn(page, loginVisits);
  });
});

test("session resilience journey: a session the auth backend rejects still signs the user out", async ({
  page,
  context,
  baseURL,
}) => {
  test.setTimeout(60_000);
  await signUpIntoLibrary(page, baseURL);

  // End the session on the server while this browser keeps its cookie, as a
  // sign-out on another device would: sign out through the auth API, then put
  // the session cookie back. The signed session cache stays gone, so the next
  // check asks the database, as it would once the cache expires.
  const sessionCookies = (await context.cookies()).filter(
    (cookie) => !/session_data/.test(cookie.name),
  );
  const signedOut = await page.evaluate(async () => {
    const response = await fetch("/api/auth/sign-out", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    return response.ok;
  });
  expect(signedOut).toBe(true);
  await context.addCookies(sessionCookies);

  await refocusUntil(page, () => /\/login/.test(page.url()));
  await expect(page).toHaveURL(/\/login\?next=%2Flibrary/);
});

test("session resilience journey: signing out in one tab signs out the others, even when the server is slow", async ({
  page,
  context,
  baseURL,
}) => {
  test.setTimeout(60_000);
  await signUpIntoLibrary(page, baseURL);
  const otherTab = await openSecondTab(context);

  // A slow sign-out: if the other tab is told before the server has expired
  // the cookies, its reload still sees the session and lands back in the app.
  await page.route("**/logout", async (route) => {
    if (
      route.request().method() !== "POST" ||
      !route.request().headers()["next-action"]
    )
      return route.fallback();
    await new Promise((resolve) => setTimeout(resolve, 3_000));
    return route.continue();
  });

  // The navbar's and the hosted plan picker's "Log out" both route here.
  await page.goto("/logout");
  await expect(page).toHaveURL(/\/login/, { timeout: 20_000 });

  await expect(otherTab).toHaveURL(/\/login\?next=%2Flibrary/, {
    timeout: 20_000,
  });
  await expect(
    otherTab.getByTestId("profile-popout-menu-trigger"),
  ).toBeHidden();
});
