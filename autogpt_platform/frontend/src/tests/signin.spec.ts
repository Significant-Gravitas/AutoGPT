// auth.spec.ts

import test from "@playwright/test";
import { getTestUser } from "./utils/auth";
import { LoginPage } from "./pages/login.page";
import { hasUrl, isVisible } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

test.beforeEach(async ({ page }) => {
  await page.goto("/login");
});

test("user can login successfully", async ({ page }) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);
  const { getId } = getSelectors(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");
  await isVisible(getId("profile-popout-menu-trigger"));
});

test("user can logout successfully", async ({ page }) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);
  const { getButton, getId } = getSelectors(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  // Open account menu
  await getId("profile-popout-menu-trigger").click();

  // Logout
  await getButton("Log out").click();
  await hasUrl(page, "/login");
});

test("login in, then out, then in again", async ({ page }) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);
  const { getButton, getId } = getSelectors(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  // Logout
  await getId("profile-popout-menu-trigger").click();
  await getButton("Log out").click();

  // Login again
  await hasUrl(page, "/login");
  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");
  await isVisible(getId("profile-popout-menu-trigger"));
});
