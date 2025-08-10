import test, { expect } from "@playwright/test";
import { LoginPage } from "./pages/login.page";
import { TEST_CREDENTIALS } from "./credentials";
import { hasFieldValue, hasUrl } from "./utils/assertion";
import { navigateToSettings, getSwitchState, TOGGLE_IDS } from "./pages/settings.page";

test.describe("Settings", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/login");
    const loginPage = new LoginPage(page);
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");
  });

  test("settings page redirects to login when not authenticated", async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto("/profile/settings");
    await hasUrl(page, /\/login/);
    await context.close();
  });

  test("user can successfully update settings", async ({ page }) => {
    const settings = await navigateToSettings(page);

    for (const id of TOGGLE_IDS) {
      const state = await getSwitchState(settings.getToggle(id));
      expect.soft(state, `Initial state of ${id}`).toBe(true);
    }

    // Turn all ON, change email/password, save
    for (const id of TOGGLE_IDS) {
      await settings.disable(id);
    }
    const tempEmail = `temp+e2e@example.com`;
    await settings.setEmail(tempEmail);
    await settings.setPassword("temporarypassword123");
    await settings.setConfirmPassword("temporarypassword123");
    await settings.saveChanges();

    // Reload and verify change email/password and OFF persisted
    await settings.goto();
    for (const id of TOGGLE_IDS) {
      const state = await getSwitchState(settings.getToggle(id));
      expect.soft(state, `Persisted OFF state of ${id}`).toBe(false);
    }
    await hasFieldValue(settings.getEmailInput(), tempEmail);
    await hasFieldValue(settings.getPasswordInput(), "temporarypassword123");

    // Restore original test email and password
    await settings.setEmail(TEST_CREDENTIALS.email);
    await settings.setPassword(TEST_CREDENTIALS.password);
    await settings.setConfirmPassword(TEST_CREDENTIALS.password);
    for (const id of TOGGLE_IDS) {
      await settings.enable(id);
    }
    await settings.saveChanges();

    // Reload and verify change email/password and ON persisted
    await settings.goto();
    for (const id of TOGGLE_IDS) {
      const state = await getSwitchState(settings.getToggle(id));
      expect.soft(state, `Persisted ON state of ${id}`).toBe(true);
    }
    await hasFieldValue(settings.getEmailInput(), TEST_CREDENTIALS.email);
    await hasFieldValue(settings.getPasswordInput(), TEST_CREDENTIALS.password);
  });

  test("user can cancel changes", async ({ page }) => {
    const settings = await navigateToSettings(page);

    const initialEmail = await settings.getEmailValue();
    await settings.setEmail("settings+cancel@example.com");

    await settings.cancelChanges();
    await hasFieldValue(settings.getEmailInput(), initialEmail);
  });

  test("settings form shows validation errors for invalid inputs", async ({ page }) => {
    const settings = await navigateToSettings(page);

    await settings.setEmail("invalid-email");
    await settings.setPassword("short");
    await settings.setConfirmPassword("short");

    await settings.saveChanges();

    await settings.expectValidationError("Invalid email");
    await settings.expectValidationError("String must contain at least 12 character(s)");
  });
});
