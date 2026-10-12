import type { Page } from "@playwright/test";
import { expect, test } from "./fixtures";
import { type JourneyUser, signUpBeforeAccountSetup } from "./journey-user";

test.describe("first login journey", () => {
  // The user's first page asks for their marketplace profile straight away,
  // and nothing promises that account setup, which creates it, has run by
  // then. The guard fails the journey on any failed request, so a 404 for
  // the profile fails it even when a retry later fills the menu in.
  test("a brand-new user's first page names them in the account menu without a failed request, and the next page works too", async ({
    page,
  }) => {
    const user = await signUpBeforeAccountSetup(page);

    await page.goto("/");
    await expect(page).toHaveURL(/\/home/);
    await expect(
      page.getByRole("textbox", { name: "Chat message input" }),
    ).toBeVisible();
    await expectAccountMenuToName(page, user);

    await page.getByRole("link", { name: "Marketplace" }).click();
    await expect(page).toHaveURL(/\/marketplace/);
    await expect(page.getByTestId("store-search-input")).toBeVisible();
    await expectAccountMenuToName(page, user);
  });
});

// Until the profile loads, the menu shows placeholders instead of who you are.
async function expectAccountMenuToName(page: Page, user: JourneyUser) {
  const trigger = page.getByTestId("profile-popout-menu-trigger");
  const menu = page.getByTestId("account-menu-popover");
  await trigger.click();
  const profileName = user.email.split("@")[0];
  await expect(menu.getByText(profileName, { exact: true })).toBeVisible();
  await expect(menu.getByText(user.email, { exact: true })).toBeVisible();
  // The trigger, not Escape: in CI, Escape sometimes left the menu open.
  await trigger.click();
  await expect(menu).toBeHidden();
}
