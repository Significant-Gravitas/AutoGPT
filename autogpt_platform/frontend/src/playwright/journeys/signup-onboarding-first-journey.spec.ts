import type { BrowserContext, Page, Response } from "@playwright/test";
import { expect, test } from "./fixtures";
import {
  authHeaders,
  newJourneyUser,
  signUpWithoutOnboarding,
} from "./journey-user";

// SECRT-3384. A signed-out visit to the app root goes `/` → `/home` →
// `/login?next=/home`, and the login page's "Sign up" link keeps that `next`.
// A new account must still go straight to onboarding: Home, the paywall and
// the out-of-credits prompt are never shown on the way.
//
// The paywall never gates a LOCAL build (CI's journeys leg), so there only
// the out-of-credits prompt and the pages visited can fail these; against a
// CLOUD build (a `!deploy` preview) the paywall check counts too.
// PaywallGate.test.tsx covers the paywall's own gate.

const PROMPTS = {
  outOfCredits: "out of automation credits",
  paywall: "Choose a plan to keep working",
};

// Every sign-up from this file claims its own client address on a local or
// CI stack, as journey-user.ts does for API sign-ups (the app caps sign-ups
// per address).
test.use({ extraHTTPHeaders: authHeaders() });

test.describe("a new account goes to onboarding first", () => {
  test("signing up from the app root lands on onboarding without Home, the paywall or the out-of-credits prompt", async ({
    page,
    context,
  }) => {
    const seen = await watchTheWayIn(context, page);

    await page.goto("/");
    await expect(page).toHaveURL(/\/login\?next=%2Fhome$/);
    await page.getByRole("link", { name: "Sign up" }).click();
    await expect(page).toHaveURL(/\/signup\?next=%2Fhome$/);

    const user = newJourneyUser();
    await page.getByLabel("Email").fill(user.email);
    await page.getByLabel("Password", { exact: true }).fill(user.password);
    await page.getByLabel("Confirm Password").fill(user.password);
    seen.startCounting();
    await page.getByRole("button", { name: "Sign up" }).click();

    await expectOnboardingStarted(page);
    seen.expectNothingOnTheWay();
  });

  test("logging in from the app root before finishing onboarding lands on onboarding", async ({
    page,
    context,
  }) => {
    const user = await signUpWithoutOnboarding(page);
    const seen = await watchTheWayIn(context, page);

    await page.goto("/");
    await expect(page).toHaveURL(/\/login\?next=%2Fhome$/);
    await page.getByLabel("Email").fill(user.email);
    await page.getByLabel("Password", { exact: true }).fill(user.password);
    seen.startCounting();
    await page.getByRole("button", { name: "Log in" }).click();

    await expectOnboardingStarted(page);
    seen.expectNothingOnTheWay();
  });

  test("opening Home directly before finishing onboarding shows neither prompt on the way to onboarding", async ({
    page,
    context,
  }) => {
    const user = newJourneyUser();
    // Signed in, account set up, onboarding not started.
    const signUp = await page.request.post("/api/auth/sign-up/email", {
      headers: authHeaders(),
      data: user,
    });
    expect(signUp.ok(), `sign-up failed: ${signUp.status()}`).toBe(true);
    const setup = await page.request.post("/api/proxy/api/auth/user");
    expect(setup.ok(), `account setup failed: ${setup.status()}`).toBe(true);

    const seen = await watchTheWayIn(context, page);
    const unanswered = await answerOnboardingCheckLast(page);
    seen.startCounting();
    await page.goto("/home");

    await expectOnboardingStarted(page);
    expect(
      unanswered,
      "Home inputs that never answered while the onboarding check was held",
    ).toEqual([]);
    // Home is where this one starts, so only the prompts are checked.
    seen.expectNoPrompts();
  });
});

const HOME_INPUTS = {
  credits: "/api/proxy/api/credits",
  autoTopUp: "/api/proxy/api/credits/auto-top-up",
  config: "/api/proxy/api/platform-config",
  subscription: "/api/proxy/api/credits/subscription",
};
// Counted from the held request, and well inside the 15 s URL assertion so a
// missing input fails by name rather than as a slow redirect.
const HOLD_LIMIT = 8_000;

// Holds the onboarding check until Home has the answers its prompts need (the
// paywall's too when payment is on), so a prompt that doesn't wait for
// onboarding opens on every run, not only when the check comes back last.
// Returns the inputs that hadn't answered when the hold gave up.
async function answerOnboardingCheckLast(page: Page) {
  const unanswered: string[] = [];
  const answered = new Map<string, Response>();
  const waiters: Array<() => void> = [];
  page.on("response", (response) => {
    const path = new URL(response.url()).pathname;
    if (!Object.values(HOME_INPUTS).includes(path) || answered.has(path)) {
      return;
    }
    response
      .finished()
      .catch(() => null)
      .then(() => {
        answered.set(path, response);
        waiters.splice(0).forEach((wake) => wake());
      });
  });
  async function neededPaths() {
    const config = answered.get(HOME_INPUTS.config);
    const body = config ? await config.json().catch(() => null) : null;
    const needed = [HOME_INPUTS.credits, HOME_INPUTS.autoTopUp];
    if (!config) needed.push(HOME_INPUTS.config);
    if (body?.payment_enabled) needed.push(HOME_INPUTS.subscription);
    return needed;
  }
  async function allAnswered(deadline: number) {
    while (Date.now() < deadline) {
      const missing = (await neededPaths()).filter((p) => !answered.has(p));
      if (missing.length === 0) return [];
      await new Promise<void>((wake) => {
        waiters.push(wake);
        setTimeout(wake, 250);
      });
    }
    return (await neededPaths()).filter((p) => !answered.has(p));
  }

  await page.route("**/api/proxy/api/onboarding/completed", async (route) => {
    const missing = await allAnswered(Date.now() + HOLD_LIMIT);
    for (const path of missing) {
      if (!unanswered.includes(path)) unanswered.push(path);
    }
    // Two frames, so React has rendered what those answers trigger.
    await route
      .request()
      .frame()
      .evaluate(
        () =>
          new Promise((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(resolve)),
          ),
      )
      .catch(() => {});
    // Not continue(): that skips the shared fixture's routes.
    await route.fallback();
  });
  return unanswered;
}

async function expectOnboardingStarted(page: Page) {
  await expect(page).toHaveURL(/\/onboarding/);
  await expect(page.getByTestId("intro-stage")).toBeVisible();
}

/**
 * Records every page the tab commits to (client-side navigations included)
 * and every time either prompt's text appears, from the moment counting
 * starts until the journey checks.
 */
async function watchTheWayIn(context: BrowserContext, page: Page) {
  const paths: string[] = [];
  const prompts = new Set<string>();
  let counting = false;

  await context.exposeBinding(
    "__reportPrompt",
    (_source, prompt: string, path: string) => {
      if (counting) prompts.add(`${prompt} on ${path}`);
    },
  );
  await context.addInitScript((watched: Record<string, string>) => {
    const report = (window as unknown as Record<string, unknown>)
      .__reportPrompt as (prompt: string, path: string) => void;
    // textContent (no layout) at most every 100 ms: reading innerText on
    // every mutation stalls the page enough to delay Playwright's network
    // events. The prompts are only in the DOM while they show.
    let scheduled = false;
    function check() {
      scheduled = false;
      const text = document.body?.textContent ?? "";
      for (const [prompt, copy] of Object.entries(watched)) {
        if (text.includes(copy)) report(prompt, window.location.pathname);
      }
    }
    function schedule() {
      if (scheduled) return;
      scheduled = true;
      setTimeout(check, 100);
    }
    new MutationObserver(schedule).observe(document, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }, PROMPTS);

  page.on("framenavigated", (frame) => {
    if (counting && frame === page.mainFrame()) {
      paths.push(new URL(frame.url()).pathname);
    }
  });

  return {
    startCounting() {
      counting = true;
    },
    expectNoPrompts() {
      expect([...prompts], "prompts shown on the way in").toEqual([]);
    },
    expectNothingOnTheWay() {
      expect(paths, "pages committed on the way in").not.toContain("/home");
      expect(paths, "pages committed on the way in").not.toContain("/");
      this.expectNoPrompts();
    },
  };
}
