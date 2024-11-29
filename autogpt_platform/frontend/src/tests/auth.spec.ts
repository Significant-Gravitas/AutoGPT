// auth.spec.ts
import { test } from "./fixtures";

test.describe("Authentication", () => {
  test("user can login successfully", async ({ page, loginPage, testUser }) => {
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");
    await test.expect(page.getByText("Monitor")).toBeVisible();
  });

  test("user can logout successfully", async ({
    page,
    loginPage,
    testUser,
  }) => {
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);

    await test.expect(page).toHaveURL("/");

    // Click on the user menu
    await page.getByRole("button", { name: "CN" }).click();
    // Click on the logout menu item
    await page.getByRole("menuitem", { name: "Log out" }).click();

    await test.expect(page).toHaveURL("/login");
  });

  test("login in, then out, then in again", async ({
    page,
    loginPage,
    testUser,
  }) => {
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await page.goto("/");
    await page.getByRole("button", { name: "CN" }).click();
    await page.getByRole("menuitem", { name: "Log out" }).click();
    await test.expect(page).toHaveURL("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");
    await test.expect(page.getByText("Monitor")).toBeVisible();
  });
});
