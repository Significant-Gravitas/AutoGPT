import { expect, test } from "./coverage-fixture";
import { completeOnboardingWizard } from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

// SECRT-3149: a brand-new user is walked over the real sidebar (chats, Team,
// Files, the wallet where billing is on), ends on the first tasks, never gets
// the walkthrough by itself again, and can replay it from the profile menu.
test("first run: a new user is walked through the real UI and can replay it", async ({
  page,
}) => {
  test.setTimeout(120000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);
  await completeOnboardingWizard(page, { role: "Engineering" });

  await expect(
    page.getByRole("dialog", { name: "Every task gets its own chat" }),
  ).toBeVisible({ timeout: 30000 });
  await expect(page.getByText("Meet your Head of AI.")).toHaveCount(0);

  for (const title of [
    "Experts take tasks off your plate",
    "Your results are saved in Files",
  ]) {
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await expect(page.getByRole("dialog", { name: title })).toBeVisible();
  }
  // The allowance step is there only where billing is on.
  const startHere = page.getByRole("dialog", { name: "Start with one task" });
  for (let step = 0; step < 2 && !(await startHere.isVisible()); step++) {
    await page.getByRole("button", { name: "Next", exact: true }).click();
  }
  await expect(startHere).toBeVisible();
  await startHere.getByRole("button", { name: /Start here/ }).click();
  await expect(page.getByTestId("first-run-tour")).toHaveCount(0);

  // Recorded for this user: a fresh load does not open it again.
  await page.reload();
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.getByTestId("first-run-tour")).toHaveCount(0);
  await expect(
    page.getByRole("textbox", { name: "Chat message input" }),
  ).toBeFocused();

  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Show me around" }).click();
  await expect(
    page.getByRole("dialog", { name: "Every task gets its own chat" }),
  ).toBeVisible();
});
