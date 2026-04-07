import { Page, expect } from "@playwright/test";

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
    "http://localhost:3000/api/proxy/api/onboarding/step?step=VISIT_COPILOT",
    { headers: { "Content-Type": "application/json" } },
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
  await page.goto(`http://localhost:3000${destination}`);
  await page.waitForLoadState("domcontentloaded", { timeout: 10000 });
}

/**
 * Walk through the full 4-step onboarding wizard in the browser.
 * Returns the data that was entered so tests can verify it was submitted.
 */
export async function completeOnboardingWizard(
  page: Page,
  options?: {
    name?: string;
    role?: string;
    painPoints?: string[];
  },
) {
  const name = options?.name ?? "TestUser";
  const role = options?.role ?? "Engineering";
  const painPoints = options?.painPoints ?? ["Research", "Reports & data"];

  // Step 1: Welcome — enter name
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible({
    timeout: 10000,
  });
  await page.getByLabel("Your first name").fill(name);
  await page.getByRole("button", { name: "Continue" }).click();

  // Step 2: Role — select a role
  await expect(page.getByText("What best describes you")).toBeVisible({
    timeout: 5000,
  });
  await page.getByText(role, { exact: false }).click();
  await page.getByRole("button", { name: "Continue" }).click();

  // Step 3: Pain points — select tasks
  await expect(page.getByText("What's eating your time?")).toBeVisible({
    timeout: 5000,
  });
  for (const point of painPoints) {
    await page.getByText(point, { exact: true }).click();
  }
  await page.getByRole("button", { name: "Launch Autopilot" }).click();

  // Step 4: Preparing — wait for animation to complete and redirect to /copilot
  await expect(page.getByText("Preparing your workspace")).toBeVisible({
    timeout: 5000,
  });
  await page.waitForURL(/\/copilot/, { timeout: 15000 });

  return { name, role, painPoints };
}
