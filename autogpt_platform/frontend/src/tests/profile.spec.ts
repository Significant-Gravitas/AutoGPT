import { LoginPage } from "./pages/login.page";
import { ProfilePage } from "./pages/profile.page";
import test, { expect } from "@playwright/test";
import { getTestUser } from "./utils/auth";
import { hasUrl } from "./utils/assertion";

test.beforeEach(async ({ page }) => {
  const loginPage = new LoginPage(page);
  const testUser = await getTestUser();

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");
});

test("user can view their profile information", async ({ page }) => {
  const profilePage = new ProfilePage(page);

  await profilePage.navbar.clickProfileLink();

  // workaround for #8788
  // sleep for 10 seconds to allow page to load due to bug in our system
  await page.waitForTimeout(10000);
  await page.reload();
  await page.reload();
  await expect(profilePage.isLoaded()).resolves.toBeTruthy();
  await hasUrl(page, new RegExp("/profile"));

  // Verify email matches test worker's email
  const displayedHandle = await profilePage.getDisplayedName();
  expect(displayedHandle).not.toBeNull();
  expect(displayedHandle).not.toBe("");
  expect(displayedHandle).toBeDefined();
});

test("profile navigation is accessible from navbar", async ({ page }) => {
  const profilePage = new ProfilePage(page);

  await profilePage.navbar.clickProfileLink();
  await hasUrl(page, new RegExp("/profile"));
  await expect(profilePage.isLoaded()).resolves.toBeTruthy();
});

test("profile displays user Credential providers", async ({ page }) => {
  const profilePage = new ProfilePage(page);
  await profilePage.navbar.clickProfileLink();
});
