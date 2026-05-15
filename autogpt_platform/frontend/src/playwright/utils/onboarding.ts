import { Page, expect } from "@playwright/test";

function resolveAppUrl(page: Page, destination: string) {
  const baseURL =
    page.url().startsWith("http://") || page.url().startsWith("https://")
      ? page.url()
      : (process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000");

  return new URL(destination, baseURL).toString();
}

/**
 * Complete the onboarding wizard via API.
 * Use this when a test needs an authenticated user who has already finished onboarding
 * (e.g., tests that navigate to marketplace, library, or build pages).
 *
 * The function sends a POST request to the onboarding completion endpoint using
 * the page's request context, which inherits the browser's auth cookies.
 */
export async function completeOnboardingViaAPI(page: Page) {
  await page.request.post(
    resolveAppUrl(page, "/api/proxy/api/onboarding/step"),
    {
      headers: { "Content-Type": "application/json" },
      params: { step: "VISIT_COPILOT" },
    },
  );
}

/**
 * Handle the onboarding redirect that occurs after login/signup.
 * If the page is on /onboarding, completes onboarding via API and navigates
 * to the given destination. If already past onboarding, does nothing.
 */
export async function skipOnboardingIfPresent(
  page: Page,
  destination: string = "/marketplace",
) {
  const url = page.url();
  if (!url.includes("/onboarding")) return;

  await completeOnboardingViaAPI(page);
  await page.goto(resolveAppUrl(page, destination));
  await page.waitForLoadState("domcontentloaded", { timeout: 10000 });
}

/**
 * Walk through the onboarding wizard in the browser. The Subscription step
 * is gated behind ENABLE_PLATFORM_PAYMENT and only walked when present.
 * Returns the data that was entered so tests can verify it was submitted.
 */
export async function completeOnboardingWizard(
  page: Page,
  options?: {
    name?: string;
    role?: string;
    painPoints?: string[];
    plan?: "pro" | "max";
  },
) {
  const name = options?.name ?? "TestUser";
  const role = options?.role ?? "Engineering";
  const painPoints = options?.painPoints ?? ["Research", "Reports & data"];
  const plan = options?.plan ?? "pro";

  // Step 1: Welcome — enter name
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible({
    timeout: 10000,
  });
  await page.getByLabel("What should I call you?").fill(name);
  await page.getByRole("button", { name: "Continue" }).click();

  // Step 2: Role — select a role (auto-advances after selection)
  await expect(page.getByText("What best describes you")).toBeVisible({
    timeout: 5000,
  });
  await page.getByText(role, { exact: false }).click();

  // Step 3: Pain points — select tasks
  await expect(page.getByText("What's eating your time?")).toBeVisible({
    timeout: 5000,
  });
  for (const point of painPoints) {
    await page.getByText(point, { exact: true }).click();
  }
  await page.getByRole("button", { name: "Continue" }).click();

  // Subscription step (only when ENABLE_PLATFORM_PAYMENT is on) — pick a
  // plan to advance. The "Team" CTA opens an external intake form and does
  // not advance, so we don't exercise it here. Race the Subscription header
  // against the Preparing header so the helper works in both flag states
  // without a fixed timeout that flakes under slow renders.
  const subscriptionHeader = page.getByText(/choose the plan that.s right/i);
  const preparingHeader = page.getByText("Preparing your workspace...", {
    exact: false,
  });
  const nextState = await Promise.race([
    subscriptionHeader
      .waitFor({ state: "visible", timeout: 10000 })
      .then(() => "subscription" as const),
    preparingHeader
      .waitFor({ state: "visible", timeout: 10000 })
      .then(() => "preparing" as const),
  ]);

  if (nextState === "subscription") {
    const planCta = plan === "max" ? "Upgrade to Max" : "Get Pro";
    await page.getByRole("button", { name: planCta }).click();
  }

  // Final step: Preparing — require the real transition state to appear first,
  // then wait for the app shell on /copilot rather than racing the redirect.
  await expect(preparingHeader).toBeVisible({ timeout: 10000 });
  await page.waitForURL(/\/copilot/, { timeout: 30000 });
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible({
    timeout: 15000,
  });

  return { name, role, painPoints, plan };
}
