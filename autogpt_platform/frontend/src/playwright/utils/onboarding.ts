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
      params: { step: "ONBOARDING_COMPLETE" },
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

export async function advanceToRoleStep(
  page: Page,
  plan: "pro" | "max" = "pro",
) {
  const role = page.getByText("What best describes you");
  const subscription = page.getByText(/choose the plan that.s right/i);
  const team = page.getByRole("heading", {
    name: "Your own team of AI experts.",
  });
  const autopilot = page.getByRole("heading", {
    name: "Meet AutoPilot, your Head of AI.",
  });

  for (let step = 0; step < 4; step++) {
    await expect(
      role.or(subscription).or(team).or(autopilot).first(),
    ).toBeVisible({ timeout: 10000 });
    if (await role.isVisible()) return;
    if (await subscription.isVisible()) {
      await page
        .getByRole("button", {
          name: plan === "max" ? "Upgrade to Max" : "Get Pro",
        })
        .click();
      await expect(subscription).toBeHidden({ timeout: 10000 });
    } else {
      const current = (await team.isVisible()) ? team : autopilot;
      await page.getByRole("button", { name: "Next", exact: true }).click();
      await expect(current).toBeHidden({ timeout: 10000 });
    }
  }
  await expect(role).toBeVisible({ timeout: 10000 });
}

export async function completeOnboardingWizard(
  page: Page,
  options?: {
    role?: string;
    painPoints?: string[];
    plan?: "pro" | "max";
  },
) {
  const role = options?.role ?? "Engineering";
  const painPoints = options?.painPoints ?? ["Research", "Reports & data"];
  const plan = options?.plan ?? "pro";

  await advanceToRoleStep(page, plan);
  await page.getByText(role, { exact: false }).click();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByText("What's eating your time?")).toBeVisible({
    timeout: 5000,
  });
  for (const point of painPoints) {
    await page.getByText(point, { exact: true }).click();
  }
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  const connect = page.getByRole("heading", {
    name: "Already paying for an AI subscription?",
  });
  const preparing = page.getByText("Preparing your workspace...", {
    exact: false,
  });
  await expect(connect.or(preparing).first()).toBeVisible({ timeout: 10000 });
  if (await connect.isVisible()) {
    await page.getByRole("button", { name: "Next", exact: true }).click();
  }

  await expect(preparing).toBeVisible({ timeout: 10000 });
  await page.waitForURL(/\/copilot/, { timeout: 30000 });
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible({
    timeout: 15000,
  });
  return { role, painPoints, plan };
}
