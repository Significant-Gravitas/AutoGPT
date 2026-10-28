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
  });

  await expectHomeLanding(page, copilotRequests);
  await expectAskWithinPhoneWidth(page);
});

async function expectHomeLanding(page: Page, copilotRequests: string[]) {
  await expect(page).toHaveURL(/\/home(\?|$)/, { timeout: 30000 });
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  await expect(page.getByText("Application error")).toHaveCount(0);
  // /copilot only redirects to /home; entering it from these pages crashed Next's router on Vercel.
  expect(copilotRequests).toEqual([]);
}

// Otto's first ask sits in a column with a narrow gutter below md; a card
// that bleeds past it is clipped at the screen edge.
async function expectAskWithinPhoneWidth(page: Page) {
  // A user fresh out of onboarding meets the first-run walkthrough first.
  const tour = page.getByTestId("first-run-tour");
  await tour.getByRole("button", { name: "Skip tour" }).click();
  await expect(tour).toHaveCount(0);
  const width = 390;
  await page.setViewportSize({ width, height: 844 });
  const ask = page.getByRole("region", { name: "One thing I need from you" });
  await expect(ask).toBeVisible({ timeout: 15000 });
  const box = await ask.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(width);
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
