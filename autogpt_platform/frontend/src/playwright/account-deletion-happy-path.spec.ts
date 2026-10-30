import type { Page } from "@playwright/test";
import { randomUUID } from "crypto";
import { expect, test } from "./coverage-fixture";
import { completeOnboardingViaAPI } from "./utils/onboarding";
import {
  generateTestEmail,
  generateTestPassword,
  signupTestUser,
} from "./utils/signup";

// The flag ships off: the frontend inlines its override at build time and the
// backend routes 404 without theirs. CI sets both in platform-fullstack-ci.yml
// and the Playwright config reads frontend/.env. A skip still fails CI, whose
// report validator rejects skipped tests.
const isAccountDeletionForcedOn = ["1", "true", "yes", "on"].includes(
  (process.env.NEXT_PUBLIC_FORCE_FLAG_SELF_SERVE_ACCOUNT_DELETION ?? "")
    .trim()
    .toLowerCase(),
);

test.skip(
  !isAccountDeletionForcedOn,
  "Set NEXT_PUBLIC_FORCE_FLAG_SELF_SERVE_ACCOUNT_DELETION=true in frontend/.env (then rebuild) and FORCE_FLAG_SELF_SERVE_ACCOUNT_DELETION=true in backend/.env",
);

test("account deletion happy path: user closes their account from Settings, then signs in and keeps it", async ({
  page,
}) => {
  // Continue waits at least 35 s after Keep for every server's cache.
  test.setTimeout(150000);

  const email = generateTestEmail();
  const password = await generateTestPassword();

  await signupTestUser(page, email, password, false);
  await completeOnboardingViaAPI(page);
  await connectFakeGitHubKey(page);

  await page.goto("/settings/account");
  await expect(
    page.getByText(
      "Close your account, disconnect your apps and cancel your plan. Everything in it is erased after 14 days.",
    ),
  ).toBeVisible({ timeout: 15000 });
  await page
    .getByRole("button", { name: "Delete account", exact: true })
    .click();

  const reasonStep = page.getByRole("dialog", {
    name: "Why are you leaving?",
  });
  await expect(reasonStep).toBeVisible();
  await reasonStep.getByRole("button", { name: "Skip" }).click();

  const confirmStep = page.getByRole("dialog", {
    name: "Delete your account?",
  });
  await expect(
    confirmStep.getByText(
      "Connected apps like GitHub are disconnected, and our access to them is revoked.",
    ),
  ).toBeVisible({ timeout: 15000 });

  const deleteButton = confirmStep.getByRole("button", {
    name: "Delete account",
    exact: true,
  });
  await expect(deleteButton).toBeDisabled();
  await confirmStep
    .getByRole("textbox", { name: `Type ${email} to confirm` })
    .fill(email);
  await expect(deleteButton).toBeEnabled();
  await deleteButton.click();

  await expect(page.getByText(/^Your account is closed\./)).toBeVisible({
    timeout: 20000,
  });
  await expect(page).toHaveURL(/\/login/, { timeout: 20000 });

  // Signing in within the 14 days shows when the account will be erased.
  await page.locator('input[type="email"]').fill(email);
  await page.locator('input[type="password"]').fill(password);
  await page.getByRole("button", { name: "Log in", exact: true }).click();

  await expect(page).toHaveURL(/\/account-closed/, { timeout: 20000 });
  await expect(
    page.getByRole("heading", { name: "Your account is closed" }),
  ).toBeVisible({ timeout: 15000 });
  await page.getByRole("button", { name: "Keep my account" }).click();
  await expect(
    page.getByRole("heading", { name: "Your account is open again" }),
  ).toBeVisible({ timeout: 15000 });
  const proceed = page.getByRole("button", { name: "Continue to AutoGPT" });
  await expect(proceed).toBeVisible({ timeout: 60000 });
  await proceed.click();

  await expect(page).not.toHaveURL(/\/account-closed/, { timeout: 20000 });
  await page.goto("/settings/account");
  await expect(
    page.getByRole("button", { name: "Delete account", exact: true }),
  ).toBeVisible({ timeout: 15000 });
});

async function connectFakeGitHubKey(page: Page) {
  const response = await page.request.post(
    "/api/proxy/api/integrations/github/credentials",
    {
      data: {
        provider: "github",
        type: "api_key",
        title: "E2E fake GitHub key",
        api_key: `e2e-fake-${randomUUID()}`,
      },
    },
  );
  expect(
    response.ok(),
    "expected the fake GitHub credential to be stored",
  ).toBe(true);
}
