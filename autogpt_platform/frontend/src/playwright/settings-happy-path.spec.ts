import { expect, test } from "./coverage-fixture";
import { LoginPage } from "./pages/login.page";
import { ProfileFormPage } from "./pages/profile-form.page";
import { SettingsPage } from "./pages/settings.page";

test("settings happy path: user can save notification preferences and keep them after reload and re-login", async ({
  page,
}) => {
  test.setTimeout(90000);

  const loginPage = new LoginPage(page);
  const settingsPage = new SettingsPage(page);

  await loginPage.loginAsSeededUser("smokeSettings");
  await settingsPage.open();

  const agentRunSwitch = settingsPage.getAgentRunNotificationsSwitch();
  // Assert the attribute exists before reading it — defaulting to "false"
  // would silently pass a regression that removes `aria-checked` entirely.
  await expect(agentRunSwitch).toHaveAttribute(
    "aria-checked",
    /^(true|false)$/,
  );
  const initialState = await agentRunSwitch.getAttribute("aria-checked");
  const expectedState = initialState === "true" ? "false" : "true";

  await agentRunSwitch.click();
  await settingsPage.savePreferences();
  await expect(agentRunSwitch).toHaveAttribute("aria-checked", expectedState);

  await page.reload();
  await settingsPage.open();
  await expect(settingsPage.getAgentRunNotificationsSwitch()).toHaveAttribute(
    "aria-checked",
    expectedState,
  );

  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Log out" }).click();
  await expect(page).toHaveURL(/\/login/);

  await loginPage.loginAsSeededUser("smokeSettings");
  await settingsPage.open();
  await expect(settingsPage.getAgentRunNotificationsSwitch()).toHaveAttribute(
    "aria-checked",
    expectedState,
  );
});

test("settings happy path: user can edit display name and keep it after refresh", async ({
  page,
}) => {
  test.setTimeout(90000);

  const loginPage = new LoginPage(page);
  const profileFormPage = new ProfileFormPage(page);
  const updatedDisplayName = `E2E Display ${Date.now()}`;

  await loginPage.loginAsSeededUser("smokeSettings");
  await page.goto("/profile");
  await expect(await profileFormPage.isLoaded()).toBe(true);

  await profileFormPage.setDisplayName(updatedDisplayName);
  await profileFormPage.saveChanges();

  await expect
    .poll(() => profileFormPage.getDisplayName(), { timeout: 15000 })
    .toBe(updatedDisplayName);

  await page.reload();
  await expect(await profileFormPage.isLoaded()).toBe(true);
  await expect
    .poll(() => profileFormPage.getDisplayName(), { timeout: 15000 })
    .toBe(updatedDisplayName);
});
