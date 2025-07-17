// auth.spec.ts

import test from "@playwright/test";
import { getTestUser } from "./utils/auth";
import { LoginPage } from "./pages/login.page";
import { hasUrl, isHidden, isVisible } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

test.beforeEach(async ({ page }) => {
  await page.goto("/login");
});

test("check the navigation when logged out", async ({ page }) => {
  const { getButton, getText, getLink } = getSelectors(page);

  // Marketplace is by default the homepage
  await page.goto("/");
  await hasUrl(page, "/marketplace");

  // Test marketplace link
  const marketplaceLink = getLink("Marketplace");
  await isVisible(marketplaceLink);
  await marketplaceLink.click();
  await hasUrl(page, "/marketplace");
  await isVisible(getText("Explore AI agents built for you by the community"));

  const loginBtn = getButton("Log In");
  await isVisible(loginBtn);
  await loginBtn.click();
  await hasUrl(page, "/login");
  await isHidden(loginBtn);
});

test("user can login successfully", async ({ page }) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);
  const { getId, getButton, getText, getRole } = getSelectors(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  const accountMenuTrigger = getId("profile-popout-menu-trigger");

  await isVisible(accountMenuTrigger);

  await accountMenuTrigger.click();
  const accountMenuPopover = getRole("dialog");
  await isVisible(accountMenuPopover);

  const username = testUser.email.split("@")[0];
  await isVisible(getText(username));

  const logoutBtn = getButton("Log out");
  await isVisible(logoutBtn);
  await logoutBtn.click();
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

  // Click on the profile menu trigger to open account menu
  await getId("profile-popout-menu-trigger").click();

  // Click the logout button in the popout menu
  await getButton("Log out").click();

  await test.expect(page).toHaveURL("/login");
  await loginPage.login(testUser.email, testUser.password);
  await test.expect(page).toHaveURL("/marketplace");
  await test
    .expect(page.getByTestId("profile-popout-menu-trigger"))
    .toBeVisible();
});
