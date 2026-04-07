import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { LoginPage } from "./pages/login.page";

function getAgentRunNotificationsSwitch(page: Page) {
  return page.getByRole("switch").nth(0);
}

async function openSettings(page: Page) {
  await page.goto("/profile/settings");
  await expect(page).toHaveURL(/\/profile\/settings/);
  await expect(
    page.getByText("Manage your account settings and preferences."),
  ).toBeVisible();
}

test("settings happy path: user can save notification preferences and keep them after reload and re-login", async ({
  page,
}) => {
  test.setTimeout(120000);

  const settingsUser = getSeededTestUser("smokeSettings");
  const loginPage = new LoginPage(page);

  await page.goto("/login");
  await loginPage.login(settingsUser.email, settingsUser.password);
  await openSettings(page);

  const agentRunSwitch = getAgentRunNotificationsSwitch(page);
  const initialState =
    (await agentRunSwitch.getAttribute("aria-checked")) ?? "false";
  const expectedState = initialState === "true" ? "false" : "true";

  await agentRunSwitch.click();
  await page.getByRole("button", { name: "Save preferences" }).click();

  await expect(
    page.getByText("Successfully updated notification preferences"),
  ).toBeVisible({ timeout: 15000 });
  await expect(agentRunSwitch).toHaveAttribute("aria-checked", expectedState);

  await page.reload();
  await openSettings(page);
  await expect(getAgentRunNotificationsSwitch(page)).toHaveAttribute(
    "aria-checked",
    expectedState,
  );

  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Log out" }).click();
  await expect(page).toHaveURL(/\/login/);

  await loginPage.login(settingsUser.email, settingsUser.password);
  await openSettings(page);
  await expect(getAgentRunNotificationsSwitch(page)).toHaveAttribute(
    "aria-checked",
    expectedState,
  );
});
