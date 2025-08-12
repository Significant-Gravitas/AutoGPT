import test, { expect } from "@playwright/test";
import { getTestUser } from "./utils/auth";
import { LoginPage } from "./pages/login.page";
import { hasUrl, isVisible } from "./utils/assertion";
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

test.describe("Settings Form", () => {
  test("should display email and notification forms", async ({ page }) => {
    const { getField, getButton, getText } = getSelectors(page);

    // Check email form elements
    await isVisible(getText("Security & Access"));
    await isVisible(getField("Email"));
    await isVisible(getButton("Reset password"));
    await isVisible(getButton("Update email"));

    // Check notification form elements
    await isVisible(getText("Notifications"));
    await isVisible(getButton("Cancel"));
    await isVisible(getButton("Save preferences"));

    // Check notification switches are present
    await isVisible(
      page.getByRole("switch", { name: /agent run notifications/i }),
    );
    await isVisible(page.getByRole("switch", { name: /zero balance alert/i }));
    await isVisible(page.getByRole("switch", { name: /daily summary/i }));
  });

  test("should have update email button disabled when email unchanged", async ({
    page,
  }) => {
    const { getButton } = getSelectors(page);

    const updateEmailButton = getButton("Update email");
    await expect(updateEmailButton).toBeDisabled();
  });

  test("should enable update email button when email is changed", async ({
    page,
  }) => {
    const { getField, getButton } = getSelectors(page);

    const emailField = getField("Email");
    const updateEmailButton = getButton("Update email");

    // Change email
    await emailField.fill("newemail@example.com");

    // Button should now be enabled
    await expect(updateEmailButton).toBeEnabled();
  });

  test("should show validation error for invalid email", async ({ page }) => {
    const { getField, getButton } = getSelectors(page);

    const emailField = getField("Email");
    const updateEmailButton = getButton("Update email");

    // Enter invalid email
    await emailField.fill("invalid-email");

    // Try to submit
    await updateEmailButton.click();

    // Should show validation error
    await isVisible(page.getByText("Please enter a valid email address"));
  });

  test("should show validation error for empty email", async ({ page }) => {
    const { getField, getButton } = getSelectors(page);

    const emailField = getField("Email");
    const updateEmailButton = getButton("Update email");

    // Clear email field
    await emailField.fill("");

    // Try to submit
    await updateEmailButton.click();

    // Should show validation error
    await isVisible(page.getByText("Email is required"));
  });

  test("should update email successfully", async ({ page }) => {
    const { getField, getButton } = getSelectors(page);

    const emailField = getField("Email");
    const updateEmailButton = getButton("Update email");

    // Change to valid email
    const newEmail = `test+${Date.now()}@example.com`;
    await emailField.fill(newEmail);

    // Submit form
    await updateEmailButton.click();

    // Should show loading state
    await isVisible(getButton("Saving..."));

    // Should show success message
    await isVisible(page.getByText("Successfully updated email"));

    // Button should be disabled again (no changes)
    await expect(updateEmailButton).toBeDisabled();
  });

  test("should navigate to reset password page", async ({ page }) => {
    const { getButton } = getSelectors(page);

    const resetPasswordButton = getButton("Reset password");
    await resetPasswordButton.click();

    // Should navigate to reset password page
    await hasUrl(page, "/reset-password");
  });

  test("should have save preferences button disabled when no changes made", async ({
    page,
  }) => {
    const { getButton } = getSelectors(page);

    const savePreferencesButton = getButton("Save preferences");
    await expect(savePreferencesButton).toBeDisabled();
  });

  test("should enable save preferences button when notification preference is changed", async ({
    page,
  }) => {
    const { getButton } = getSelectors(page);

    const agentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    const savePreferencesButton = getButton("Save preferences");

    // Toggle notification switch
    await agentRunSwitch.click();

    // Button should now be enabled
    await expect(savePreferencesButton).toBeEnabled();
  });

  test("should toggle notification switches correctly", async ({ page }) => {
    const switches = [
      page.getByRole("switch", { name: /agent run notifications/i }),
      page.getByRole("switch", { name: /block execution failures/i }),
      page.getByRole("switch", { name: /continuous agent errors/i }),
      page.getByRole("switch", { name: /zero balance alert/i }),
      page.getByRole("switch", { name: /low balance warning/i }),
      page.getByRole("switch", { name: /daily summary/i }),
      page.getByRole("switch", { name: /weekly summary/i }),
      page.getByRole("switch", { name: /monthly summary/i }),
    ];

    for (const switchElement of switches) {
      // Get initial state
      const initialState = await switchElement.isChecked();

      // Toggle switch
      await switchElement.click();

      // Verify state changed
      const newState = await switchElement.isChecked();
      expect(newState).toBe(!initialState);
    }
  });

  test("should save notification preferences successfully", async ({
    page,
  }) => {
    const { getButton } = getSelectors(page);

    const agentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    const savePreferencesButton = getButton("Save preferences");

    // Toggle a notification switch
    await agentRunSwitch.click();

    // Save preferences
    await savePreferencesButton.click();

    // Should show loading state
    await isVisible(getButton("Saving..."));

    // Should show success message
    await isVisible(
      page.getByText("Successfully updated notification preferences"),
    );

    // Button should be disabled again (no changes)
    await expect(savePreferencesButton).toBeDisabled();
  });

  test("should cancel notification changes", async ({ page }) => {
    const { getButton } = getSelectors(page);

    const agentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    const cancelButton = getButton("Cancel");
    const savePreferencesButton = getButton("Save preferences");

    // Get initial state
    const initialState = await agentRunSwitch.isChecked();

    // Toggle switch
    await agentRunSwitch.click();

    // Verify state changed and button is enabled
    expect(await agentRunSwitch.isChecked()).toBe(!initialState);
    await expect(savePreferencesButton).toBeEnabled();

    // Cancel changes
    await cancelButton.click();

    // Should revert to initial state
    expect(await agentRunSwitch.isChecked()).toBe(initialState);

    // Button should be disabled again
    await expect(savePreferencesButton).toBeDisabled();
  });

  test("should maintain separate form states", async ({ page }) => {
    const { getField, getButton } = getSelectors(page);

    const emailField = getField("Email");
    const updateEmailButton = getButton("Update email");
    const agentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    const savePreferencesButton = getButton("Save preferences");

    // Change email
    await emailField.fill("newemail@example.com");
    await expect(updateEmailButton).toBeEnabled();
    await expect(savePreferencesButton).toBeDisabled();

    // Change notification
    await agentRunSwitch.click();
    await expect(updateEmailButton).toBeEnabled();
    await expect(savePreferencesButton).toBeEnabled();

    // Submit email form
    await updateEmailButton.click();
    await isVisible(page.getByText("Successfully updated email"));

    // Email button should be disabled, notification button still enabled
    await expect(updateEmailButton).toBeDisabled();
    await expect(savePreferencesButton).toBeEnabled();
  });

  test("should handle multiple notification changes in sequence", async ({
    page,
  }) => {
    const { getButton } = getSelectors(page);

    const switches = [
      page.getByRole("switch", { name: /agent run notifications/i }),
      page.getByRole("switch", { name: /zero balance alert/i }),
      page.getByRole("switch", { name: /daily summary/i }),
    ];
    const savePreferencesButton = getButton("Save preferences");

    // Toggle multiple switches
    for (const switchElement of switches) {
      await switchElement.click();
    }

    // Save all changes
    await savePreferencesButton.click();
    await isVisible(
      page.getByText("Successfully updated notification preferences"),
    );

    // All switches should maintain their new states
    // (This tests that the form properly handles multiple simultaneous changes)
    await expect(savePreferencesButton).toBeDisabled();
  });

  test("should show proper loading states during submission", async ({
    page,
  }) => {
    const { getField, getButton } = getSelectors(page);

    // Test email form loading state
    const emailField = getField("Email");
    const updateEmailButton = getButton("Update email");

    await emailField.fill("loading-test@example.com");
    await updateEmailButton.click();

    // Should show loading state briefly
    await isVisible(getButton("Saving..."));

    // Wait for completion
    await isVisible(page.getByText("Successfully updated email"));

    // Test notification form loading state
    const agentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    const savePreferencesButton = getButton("Save preferences");

    await agentRunSwitch.click();
    await savePreferencesButton.click();

    // Should show loading state briefly
    await isVisible(getButton("Saving..."));

    // Wait for completion
    await isVisible(
      page.getByText("Successfully updated notification preferences"),
    );
  });

  test("should persist form data on page reload", async ({ page }) => {
    const { getButton } = getSelectors(page);

    const agentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    const savePreferencesButton = getButton("Save preferences");

    // Get initial state
    const initialState = await agentRunSwitch.isChecked();

    // Toggle and save
    await agentRunSwitch.click();
    await savePreferencesButton.click();
    await isVisible(
      page.getByText("Successfully updated notification preferences"),
    );

    // Reload page
    await page.reload();
    await hasUrl(page, "/profile/settings");

    // Switch should maintain the changed state
    const newAgentRunSwitch = page.getByRole("switch", {
      name: /agent run notifications/i,
    });
    expect(await newAgentRunSwitch.isChecked()).toBe(!initialState);
  });
});
