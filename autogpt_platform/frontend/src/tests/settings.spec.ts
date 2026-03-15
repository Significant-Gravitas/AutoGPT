import test, { expect } from "@playwright/test";
import { getTestUser } from "./utils/auth";
import { LoginPage } from "./pages/login.page";
import { hasAttribute, hasUrl, isHidden, isVisible } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

test.beforeEach(async ({ page }) => {
  const testUser = await getTestUser();
  const loginPage = new LoginPage(page);

  // Login and navigate to settings
  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await hasUrl(page, "/marketplace");

  // Navigate to settings page
  await page.goto("/profile/settings");
  await hasUrl(page, "/profile/settings");
});

test("should display email form elements correctly", async ({ page }) => {
  const { getField, getButton, getText, getLink } = getSelectors(page);

  // Check email form elements are displayed
  await isVisible(getText("Security & Access"));
  await isVisible(getField("Email"));
  await isVisible(getLink("Reset password"));
  await isVisible(getButton("Update email"));

  const updateEmailButton = getButton("Update email");
  const resetPasswordButton = getLink("Reset password");

  // Button should be disabled initially (no changes)
  await expect(updateEmailButton).toBeDisabled();

  // Test reset password navigation
  await hasAttribute(resetPasswordButton, "href", "/reset-password");
});

test("should show validation error for empty email", async ({ page }) => {
  const { getField, getButton } = getSelectors(page);

  const emailField = getField("Email");
  const updateEmailButton = getButton("Update email");

  await emailField.fill("");
  await updateEmailButton.click();
  await isVisible(page.getByText("Email is required"));
});

test("should show validation error for invalid email", async ({ page }) => {
  const { getField, getButton } = getSelectors(page);

  const emailField = getField("Email");
  const updateEmailButton = getButton("Update email");

  await emailField.fill("invalid email");
  await updateEmailButton.click();
  await isVisible(page.getByText("Please enter a valid email address"));
});

test("should handle valid email", async ({ page }) => {
  const { getField, getButton } = getSelectors(page);

  const emailField = getField("Email");
  const updateEmailButton = getButton("Update email");

  // Test successful email update
  const newEmail = `test+${Date.now()}@example.com`;
  await emailField.fill(newEmail);
  await expect(updateEmailButton).toBeEnabled();
  await updateEmailButton.click();
  await isHidden(page.getByText("Email is required"));
  await isHidden(page.getByText("Please enter a valid email address"));
});

test("should handle complete notification form functionality and form interactions", async ({
  page,
}) => {
  const { getButton } = getSelectors(page);

  // Check notification form elements are displayed
  await isVisible(
    page.getByRole("heading", { name: "Notifications", exact: true }),
  );

  await isVisible(getButton("Cancel"));
  await isVisible(getButton("Save preferences"));

  // Check all notification switches are present - get all switches on page
  const switches = await page.getByRole("switch").all();

  for (const switchElement of switches) {
    await isVisible(switchElement);
  }

  const savePreferencesButton = getButton("Save preferences");
  const cancelButton = getButton("Cancel");

  // Button should be disabled initially (no changes)
  await expect(savePreferencesButton).toBeDisabled();

  // Test switch toggling functionality
  for (const switchElement of switches) {
    const initialState = await switchElement.isChecked();
    await switchElement.click();
    const newState = await switchElement.isChecked();
    expect(newState).toBe(!initialState);
  }

  // Test button enabling when changes are made
  if (switches.length > 0) {
    await expect(savePreferencesButton).toBeEnabled();
  }

  // Test cancel functionality
  await cancelButton.click();
  // Wait for form state to update after cancel
  await page.waitForTimeout(100);
  await expect(savePreferencesButton).toBeDisabled();

  // Test successful save with multiple switches
  const testSwitches = switches.slice(0, Math.min(3, switches.length));
  for (const switchElement of testSwitches) {
    await switchElement.click();
  }
  await expect(savePreferencesButton).toBeEnabled();
  await savePreferencesButton.click();
  await isVisible(getButton("Saving..."));
  await isVisible(
    page.getByText("Successfully updated notification preferences"),
  );

  // Test persistence after page reload
  if (testSwitches.length > 0) {
    const finalState = await testSwitches[0].isChecked();
    await page.reload();
    await hasUrl(page, "/profile/settings");
    const reloadedSwitches = await page.getByRole("switch").all();
    if (reloadedSwitches.length > 0) {
      expect(await reloadedSwitches[0].isChecked()).toBe(finalState);
    }
  }
});
