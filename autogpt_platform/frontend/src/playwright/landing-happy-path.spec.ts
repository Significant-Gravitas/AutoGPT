import type { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { LoginPage } from "./pages/login.page";
import { completeOnboardingWizard } from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

test("landing happy path: a signed-in user opening /login lands on /home", async ({
  page,
}) => {
  test.setTimeout(60000);

  const copilotRequests = recordCopilotNavigations(page);
  const testUser = getSeededTestUser("smokeAuth");
  await page.goto("/login");
  await new LoginPage(page).login(testUser.email, testUser.password);

  await page.goto("/login");

  await expectHomeLanding(page, copilotRequests);
});

test("landing happy path: a user finishing onboarding lands on /home", async ({
  page,
}) => {
  test.setTimeout(60000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);

  const copilotRequests = recordCopilotNavigations(page);
  await completeOnboardingWizard(page, {
    role: "Engineering",
    painPoints: ["Research", "Reports & data"],
  });

  await expectHomeLanding(page, copilotRequests);
});

async function expectHomeLanding(page: Page, copilotRequests: string[]) {
  await expect(page).toHaveURL(/\/home(\?|$)/, { timeout: 30000 });
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  await expect(page.getByText("Application error")).toHaveCount(0);
  // /copilot only redirects to /home; entering it from these pages crashed Next's router on Vercel.
  expect(copilotRequests).toEqual([]);
}

function recordCopilotNavigations(page: Page): string[] {
  const requests: string[] = [];
  page.on("request", (request) => {
    const headers = request.headers();
    if (
      new URL(request.url()).pathname === "/copilot" &&
      !headers["next-router-prefetch"] &&
      !headers["next-router-segment-prefetch"]
    ) {
      requests.push(request.url());
    }
  });
  return requests;
}
