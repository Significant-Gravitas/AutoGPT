// auth.spec.ts

import test from "@playwright/test";
import { BuildPage } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import { hasUrl, isHidden, isVisible } from "./utils/assertion";
import { getTestUser } from "./utils/auth";
import { getSelectors } from "./utils/selectors";

test.beforeEach(async ({ page }) => {
  await page.goto("/login");
});

test("check the navigation when logged out", async ({ page }) => {
  const { getButton, getText, getLink } = getSelectors(page);

  // Test marketplace link
  const marketplaceLink = getLink("Marketplace");
  await isVisible(marketplaceLink);
  await marketplaceLink.click();
  await hasUrl(page, "/marketplace");
  await isVisible(getText("Explore AI agents", { exact: false }));

  // Test login button
  const loginBtn = getButton("Log In");
  await isVisible(loginBtn);
  await loginBtn.click();
  await hasUrl(page, "/login");
  await isHidden(loginBtn);
});

test("user can login successfully", async ({ page }) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);
  const { getId, getButton, getRole } = getSelectors(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  const accountMenuTrigger = getId("profile-popout-menu-trigger");

  await isVisible(accountMenuTrigger);

  await accountMenuTrigger.click();
  const accountMenuPopover = getRole("dialog");
  await isVisible(accountMenuPopover);

  const accountMenuUserEmail = getId("account-menu-user-email");
  await isVisible(accountMenuUserEmail);
  await test
    .expect(accountMenuUserEmail)
    .toHaveText(testUser.email.split("@")[0].toLowerCase());

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

test("multi-tab logout with WebSocket cleanup", async ({ context }) => {
  const testUser = await getTestUser();

  // Tab 1
  const page1 = await context.newPage();
  const builderPage1 = new BuildPage(page1);

  // Capture console errors to ensure WebSocket cleanup prevents errors
  const consoleErrors: string[] = [];
  page1.on("console", (msg) => {
    if (msg.type() === "error" && msg.text().includes("WebSocket")) {
      consoleErrors.push(`Page1: ${msg.text()}`);
    }
  });

  const loginPage1 = new LoginPage(page1);
  const { getButton: getButton1, getId: getId1 } = getSelectors(page1);

  // Login
  await page1.goto("/login");
  await loginPage1.login(testUser.email, testUser.password);
  await hasUrl(page1, "/marketplace");

  //  Navigate to builder + wait for WebSocket connection
  await page1.goto("/build");
  await hasUrl(page1, "/build");
  await builderPage1.closeTutorial();
  await page1.waitForTimeout(1000);
  await isVisible(getId1("profile-popout-menu-trigger"));

  // Tab 2
  const page2 = await context.newPage();

  const { getId: getId2 } = getSelectors(page2);

  page2.on("console", (msg) => {
    if (msg.type() === "error" && msg.text().includes("WebSocket")) {
      consoleErrors.push(`Page2: ${msg.text()}`);
    }
  });

  // Navigate to builder + wait for WebSocket connection
  await page2.goto("/build");
  await hasUrl(page2, "/build");
  await page2.waitForTimeout(1000);
  await isVisible(getId2("profile-popout-menu-trigger"));

  // Tab 1: Logout
  await getId1("profile-popout-menu-trigger").click();
  await getButton1("Log out").click();
  await hasUrl(page1, "/login");

  // Tab 2: Wait for cross-tab logout to take effect and check if redirected to login
  await page2.waitForTimeout(2000); // Give time for cross-tab logout mechanism

  // Check if Tab 2 has been redirected to login or refresh the page to trigger redirect
  try {
    await page2.reload();
    await hasUrl(page2, "/login?next=%2Fbuild");
  } catch {
    // If reload fails, the page might already be redirecting
    await hasUrl(page2, "/login?next=%2Fbuild");
  }

  // Verify the profile menu is no longer visible (user is logged out)
  await isHidden(getId2("profile-popout-menu-trigger"));

  // Verify no WebSocket connection errors occurred during logout
  test.expect(consoleErrors).toHaveLength(0);
  if (consoleErrors.length > 0) {
    console.log("WebSocket errors during logout:", consoleErrors);
  }

  // Clean up
  await page1.close();
  await page2.close();
});

test("logged in user is redirected from /login to /library", async ({
  page,
}) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  await page.goto("/login");
  await hasUrl(page, "/library?sort=updatedAt");
});

test("logged in user is redirected from /signup to /library", async ({
  page,
}) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);

  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  await page.goto("/signup");
  await hasUrl(page, "/library?sort=updatedAt");
});
