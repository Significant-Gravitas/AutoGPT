import { test, expect } from "./fixtures";

test.describe("Authentication", () => {
  test("user can login successfully", async ({ page, loginPage, testUser }) => {
    await page.goto("/login"); // Make sure we're on the login page
    await loginPage.login(testUser.email, testUser.password);
    // expect to be redirected to the home page
    await expect(page).toHaveURL("/");
    // expect to see the Monitor text
    await expect(page.getByText("Monitor")).toBeVisible();
  });

  test("user can logout successfully", async ({
    page,
    loginPage,
    testUser,
  }) => {
    await page.goto("/login"); // Make sure we're on the login page
    await loginPage.login(testUser.email, testUser.password);

    // Expect to be on the home page
    await expect(page).toHaveURL("/");
    // Click on the user menu
    await page.getByRole("button", { name: "CN" }).click();
    // Click on the logout menu item
    await page.getByRole("menuitem", { name: "Log out" }).click();
    // Expect to be redirected to the login page
    await expect(page).toHaveURL("/login");
  });

  test("login in, then out, then in again", async ({
    page,
    loginPage,
    testUser,
  }) => {
    await page.goto("/login"); // Make sure we're on the login page
    await loginPage.login(testUser.email, testUser.password);
    await page.goto("/");
    await page.getByRole("button", { name: "CN" }).click();
    await page.getByRole("menuitem", { name: "Log out" }).click();
    await expect(page).toHaveURL("/login");
    await loginPage.login(testUser.email, testUser.password);
    await expect(page).toHaveURL("/");
    await expect(page.getByText("Monitor")).toBeVisible();
  });
});
