import { expect, test } from "./coverage-fixture";
import { LoginPage } from "./pages/login.page";
import { SettingsPage } from "./pages/settings.page";

test("settings happy path: user can save notification preferences and keep them after reload and re-login", async ({
  page,
}) => {
  test.setTimeout(120000);

  const loginPage = new LoginPage(page);
  const settingsPage = new SettingsPage(page);

  await loginPage.loginAsSeededUser("smokeSettings");
  await settingsPage.open();

  const agentRunSwitch = settingsPage.getAgentRunNotificationsSwitch();
  const initialState =
    (await agentRunSwitch.getAttribute("aria-checked")) ?? "false";
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
