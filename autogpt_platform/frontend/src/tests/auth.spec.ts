// auth.spec.ts
import { test } from "./fixtures";

test.describe("Authentication", () => {
  test("user can login successfully", async ({ page, loginPage, testUser }) => {
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/marketplace");
    await test
      .expect(page.getByTestId("profile-popout-menu-trigger"))
      .toBeVisible();
  });

  test("user can logout successfully", async ({
    page,
    loginPage,
    testUser,
  }) => {
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);

    await test.expect(page).toHaveURL("/marketplace");

    // Click on the profile menu trigger to open popout
    await page.getByTestId("profile-popout-menu-trigger").click();

    // Wait for menu to be visible before clicking logout
    await page.getByRole("button", { name: "Log out" }).waitFor({
      state: "visible",
      timeout: 5000,
    });

    // Click the logout button in the popout menu
    await page.getByRole("button", { name: "Log out" }).click();

    await test.expect(page).toHaveURL("/login");
  });

  test("login in, then out, then in again", async ({
    page,
    loginPage,
    testUser,
  }) => {
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/marketplace");
    // Click on the profile menu trigger to open popout
    await page.getByTestId("profile-popout-menu-trigger").click();

    // Click the logout button in the popout menu
    await page.getByRole("button", { name: "Log out" }).click();

    await test.expect(page).toHaveURL("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/marketplace");
    await test
      .expect(page.getByTestId("profile-popout-menu-trigger"))
      .toBeVisible();
  });
});
